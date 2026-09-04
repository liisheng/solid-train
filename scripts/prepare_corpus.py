"""Run the restartable production corpus pipeline over pinned source streams."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.benchmark_index import BenchmarkIndex  # noqa: E402
from tinybench_lm.corpus_pipeline import (  # noqa: E402
    CorpusPipelineError,
    CorpusState,
    StreamedSourceRow,
    assert_write_space,
    iter_huggingface_source,
    load_acquisition_protocol,
)
from tinybench_lm.data_protocols import load_decontamination_protocol  # noqa: E402
from tinybench_lm.source_manifest import FINAL_TOKEN_COUNTER_ID, load_source_registry  # noqa: E402
from tinybench_lm.tokenizer import load_tokenizer_artifact  # noqa: E402

PHYSICAL_SOURCE_ORDER = (
    "fineweb_edu",
    "dclm",
    "openwebmath",
    "narrative",
    "reserved_textbook",
    "reserved_wikipedia",
)
STAGES = ("ingest", "dedup", "benchmark-index", "decontaminate", "assign", "publish", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--state", type=Path, required=True, help="restartable SQLite corpus state")
    parser.add_argument("--cache-dir", type=Path, required=True, help="explicit Hugging Face cache on the data volume")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPOSITORY_ROOT / "data" / "tokenizer_final")
    parser.add_argument(
        "--benchmark-items",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "decontamination" / "benchmark_items.jsonl",
    )
    parser.add_argument("--benchmark-index", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--target-fraction", type=float, required=True, help="0 < fraction <= 1; use 0.01 for the first slice")
    parser.add_argument("--pool-factor", type=float, default=1.30, help="slice acquisition headroom before filtering")
    parser.add_argument("--source", action="append", choices=PHYSICAL_SOURCE_ORDER, help="limit ingestion to named physical sources")
    parser.add_argument("--confirm-full-scan", action="store_true", help="required when target-fraction is 1")
    return parser.parse_args()


def physical_target_tokens(source_id: str, registry: dict, acquisition: dict) -> int:
    stable = {str(item["source_id"]): int(item["target_tokens_at_11b"]) for item in registry["stable_sources"]}
    reserved = {
        str(item["source_id"]): int(item["target_tokens_at_minimum"])
        for item in registry["reserved_sources"]
    }
    validation_total = sum(int(value) for value in acquisition["validation"]["targets"].values())
    shares = acquisition["validation"]["stable_source_shares"]
    if source_id == "fineweb_edu":
        return stable[source_id] + reserved["reserved_science"] + reserved["reserved_edu_decile"] + round(validation_total * float(shares[source_id]))
    if source_id == "openwebmath":
        return stable[source_id] + reserved["reserved_math_prose"] + round(validation_total * float(shares[source_id]))
    if source_id in stable:
        return stable[source_id] + round(validation_total * float(shares[source_id]))
    return reserved[source_id]


def bounded_rows(
    rows: Iterable[StreamedSourceRow],
    *,
    token_counter,
    token_budget: int,
) -> Iterator[StreamedSourceRow]:
    consumed = 0
    for row in rows:
        yield row
        consumed += int(token_counter(row.candidate.text))
        if consumed >= token_budget:
            break


def remaining_slice_budget(token_budget: int, consumed_tokens: int) -> int:
    """Return the persisted remainder so a resumed slice cannot acquire a second budget."""
    if token_budget < 0 or consumed_tokens < 0:
        raise ValueError("token budgets and consumed tokens must be nonnegative")
    return max(0, token_budget - consumed_tokens)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    if temporary.exists():
        raise CorpusPipelineError(f"stale evidence staging file exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 0 < args.target_fraction <= 1:
        raise SystemExit("--target-fraction must be in (0, 1]")
    if args.pool_factor < 1:
        raise SystemExit("--pool-factor must be at least 1")
    if args.target_fraction == 1 and not args.confirm_full_scan and args.stage in {"ingest", "all"}:
        raise SystemExit("the full source scan requires --confirm-full-scan after slice forecasts are reviewed")
    acquisition = load_acquisition_protocol()
    registry = load_source_registry()
    decontamination = load_decontamination_protocol(
        REPOSITORY_ROOT / "configs" / "data" / "decontam_v2.yaml"
    )
    tokenizer, _ = load_tokenizer_artifact(args.tokenizer_dir)
    token_counter = lambda text: len(tokenizer.encode(text).ids)
    assert_write_space(args.state.parent, acquisition)
    assert_write_space(args.cache_dir, acquisition)
    selection_results = ()
    accepted_hash = None
    accepted_rows = None
    decisions_hash = None
    decision_rows = None
    stage_metrics: dict[str, dict] = {}
    stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    with CorpusState(
        args.state,
        token_counter=token_counter,
        token_counter_id=FINAL_TOKEN_COUNTER_ID,
        acquisition=acquisition,
        registry=registry,
    ) as state:
        if "ingest" in stages:
            stage_started = time.perf_counter()
            source_metrics: dict[str, dict[str, int | float | bool]] = {}
            for source_id in args.source or PHYSICAL_SOURCE_ORDER:
                cursor = state.cursor(source_id)
                source_started = time.perf_counter()
                stream: Iterable[StreamedSourceRow] = iter_huggingface_source(
                    source_id,
                    cache_dir=args.cache_dir,
                    start_row_index=cursor.next_row_index,
                    registry=registry,
                )
                full_scan = args.target_fraction == 1
                remaining_budget = 0
                if not full_scan:
                    budget = int(math.ceil(physical_target_tokens(source_id, registry, acquisition) * args.target_fraction * args.pool_factor))
                    remaining_budget = remaining_slice_budget(budget, cursor.consumed_tokens)
                    stream = bounded_rows(stream, token_counter=token_counter, token_budget=remaining_budget)
                candidates, scores = itertools.tee(stream)
                written = 0
                if full_scan or remaining_budget > 0:
                    written = state.ingest(
                        source_id,
                        (row.candidate for row in candidates),
                        start_row_index=cursor.next_row_index,
                        scores=(row.score for row in scores),
                        mark_complete=full_scan,
                        allow_empty_completion=full_scan,
                    )
                updated_cursor = state.cursor(source_id)
                source_metrics[source_id] = {
                    "elapsed_seconds": round(time.perf_counter() - source_started, 6),
                    "documents_written": written,
                    "next_row_index": updated_cursor.next_row_index,
                    "consumed_tokens": updated_cursor.consumed_tokens,
                    "source_complete": updated_cursor.complete,
                }
            stage_metrics["ingest"] = {
                "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
                "sources": source_metrics,
            }
        if "dedup" in stages:
            stage_started = time.perf_counter()
            processed = state.run_deduplication()
            stage_metrics["dedup"] = {
                "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
                "documents_processed": processed,
            }
        if "benchmark-index" in stages:
            stage_started = time.perf_counter()
            with BenchmarkIndex(args.benchmark_index, protocol=decontamination) as index:
                index_summary = index.build(
                    args.benchmark_items,
                    expected_sha256=str(acquisition["decontamination"]["benchmark_items_sha256"]),
                )
            stage_metrics["benchmark-index"] = {
                "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
                "items": index_summary.items,
                "texts": index_summary.texts,
                "complete": index_summary.complete,
            }
        if "decontaminate" in stages:
            stage_started = time.perf_counter()
            with BenchmarkIndex(args.benchmark_index, protocol=decontamination) as index:
                processed = state.run_decontamination(index)
            stage_metrics["decontaminate"] = {
                "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
                "documents_processed": processed,
            }
        if "assign" in stages:
            stage_started = time.perf_counter()
            selection_results = state.assign_production(target_fraction=args.target_fraction)
            incomplete = [result for result in selection_results if not result.complete]
            if incomplete:
                detail = ", ".join(
                    f"{item.selection_id}={item.selected_tokens}/{item.target_tokens}" for item in incomplete
                )
                raise CorpusPipelineError(f"selection targets are incomplete: {detail}")
            stage_metrics["assign"] = {
                "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
                "selected_documents": sum(item.selected_documents for item in selection_results),
                "selected_tokens": sum(item.selected_tokens for item in selection_results),
            }
        if "publish" in stages:
            stage_started = time.perf_counter()
            accepted_rows, accepted_hash, decision_rows, decisions_hash = state.publish_jsonl_bundle(
                args.accepted_output, args.decisions_output
            )
            stage_metrics["publish"] = {
                "elapsed_seconds": round(time.perf_counter() - stage_started, 6),
                "accepted_rows": accepted_rows,
                "decision_rows": decision_rows,
                "accepted_bytes": args.accepted_output.stat().st_size,
                "decision_bytes": args.decisions_output.stat().st_size,
            }
        summary = state.summary()
        isolation = state.isolation_evidence()
        dedup_classified = int(state.connection.execute("SELECT COUNT(*) FROM dedup_decisions").fetchone()[0])
        decontamination_classified = int(state.connection.execute("SELECT COUNT(*) FROM decontamination").fetchone()[0])
        coverage = {
            "filter_accepted": summary.filter_accepted,
            "dedup_classified": dedup_classified,
            "dedup_kept": summary.dedup_kept,
            "decontamination_classified": decontamination_classified,
        }
        coverage_complete = (
            dedup_classified == summary.filter_accepted
            and decontamination_classified == dedup_classified
        )
        required = {"ingest", "dedup", "benchmark-index", "decontaminate", "assign", "publish"}
        complete_run = required.issubset(stages) and accepted_hash is not None and decisions_hash is not None and all(
            item.complete for item in selection_results
        ) and coverage_complete and isolation["status"] == "PASS"
        state.connection.commit()
        state.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        state_sha256 = hashlib.sha256(args.state.read_bytes()).hexdigest()
        evidence = {
            "status": "PASS" if complete_run else "NOT_RUN",
            "scale": "FULL" if args.target_fraction == 1 else "SLICE",
            "target_fraction": args.target_fraction,
            "stage_metrics": stage_metrics,
            "summary": summary.to_dict(),
            "isolation": isolation,
            "coverage": {**coverage, "status": "PASS" if coverage_complete else "NOT_RUN"},
            "selection": [
                {
                    "selection_id": item.selection_id,
                    "target_tokens": item.target_tokens,
                    "selected_tokens": item.selected_tokens,
                    "selected_documents": item.selected_documents,
                    "status": "PASS" if item.complete else "FAIL",
                }
                for item in selection_results
            ],
            "accepted_output": str(args.accepted_output) if accepted_hash else None,
            "accepted_rows": accepted_rows,
            "accepted_sha256": accepted_hash,
            "decisions_output": str(args.decisions_output) if decisions_hash else None,
            "decision_rows": decision_rows,
            "decisions_sha256": decisions_hash,
            "sqlite_state": str(args.state),
            "sqlite_state_sha256": state_sha256,
            "protocol_digests": {
                "acquisition": acquisition["_digest"],
                "sources": registry["_digest"],
                "decontamination": decontamination["_digest"],
            },
            "token_counter_id": FINAL_TOKEN_COUNTER_ID,
        }
        write_json_atomic(args.evidence_output, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
