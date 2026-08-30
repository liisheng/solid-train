"""Build and verify the final 12,288-ID tokenizer contract (Plan Section 5.3).

Modes:

    --mode plan      print the deterministic stratified sample plan. Reads no documents.
    --mode fixture   build a FIXTURE-scope tokenizer from a local JSONL corpus and verify it.
    --mode final     build the real tokenizer. Gated, and currently fails closed because no
                     stable source revision is pinned and no 2 GB sample has been drawn.
    --mode verify    verify an existing artifact directory against the frozen contract.

The fixture corpus is JSONL with one object per line:

    {"source_id": "fineweb_edu", "document_id": "doc-0001", "text": "..."}

`source_id` must be one of the frozen stable sources in configs/data/sources_v1.yaml.
This script never downloads a corpus and never trains the 2 GB tokenizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.tokenizer import (  # noqa: E402
    SCOPE_FINAL,
    SCOPE_FIXTURE,
    SampleDocument,
    TokenizerNotReadyError,
    assert_ready_for_final_tokenizer_build,
    build_fixture_tokenizer,
    build_sample_plan,
    load_tokenizer_artifact,
    load_tokenizer_protocol,
    verify_tokenizer,
    write_tokenizer_artifact,
)

EXIT_OK = 0
EXIT_FAILED_VERIFICATION = 1
EXIT_BLOCKED = 2


def read_corpus(path: Path) -> list[SampleDocument]:
    documents: list[SampleDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            for field in ("source_id", "document_id", "text"):
                if field not in payload:
                    raise ValueError(f"{path}:{line_number} is missing {field!r}")
            documents.append(
                SampleDocument(
                    source_id=str(payload["source_id"]),
                    document_id=str(payload["document_id"]),
                    text=str(payload["text"]),
                )
            )
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return documents


def print_report(report) -> None:
    width = max((len(result.check_id) for result in report.results), default=0)
    for result in report.results:
        print(f"{result.status:<8} {result.check_id:<{width}}  {result.observed}  # {result.reason}")
    print()
    print("Facts (informational): " + json.dumps(report.facts, sort_keys=True))
    print("RESULT: " + ("PASS" if report.ok else "FAIL"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and verify the final tokenizer contract",
        epilog="The real 2 GB build is DEFERRED until stable source revisions are pinned.",
    )
    parser.add_argument("--mode", choices=["plan", "fixture", "final", "verify"], default="plan")
    parser.add_argument("--corpus", type=Path, help="JSONL fixture corpus for --mode fixture/final")
    parser.add_argument("--output-dir", type=Path, help="artifact directory to write or verify")
    parser.add_argument(
        "--represented-bytes",
        type=int,
        default=None,
        help="override the represented sample size; the frozen default is 2 GiB",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_tokenizer_protocol()

    if args.mode == "plan":
        plan = build_sample_plan(represented_bytes=args.represented_bytes, protocol=protocol)
        payload = plan.to_dict()
        payload["sample_plan_digest"] = plan.digest
        payload["final_2gb_build"] = dict(protocol["readiness"]["final_2gb_build"])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK

    if args.mode == "verify":
        if args.output_dir is None:
            raise SystemExit("--mode verify requires --output-dir")
        tokenizer, record = load_tokenizer_artifact(args.output_dir)
        print(f"build_scope: {record.get('build_scope', 'UNKNOWN')}")
        print(f"sample_plan_digest: {record.get('sample_plan_digest', 'UNKNOWN')}")
        report = verify_tokenizer(tokenizer, protocol=protocol)
        print_report(report)
        return EXIT_OK if report.ok else EXIT_FAILED_VERIFICATION

    if args.corpus is None:
        raise SystemExit(f"--mode {args.mode} requires --corpus")

    if args.mode == "final":
        try:
            assert_ready_for_final_tokenizer_build(protocol)
        except TokenizerNotReadyError as error:
            print(f"BLOCKED {error}", file=sys.stderr)
            print("No tokenizer was trained. Nothing was written.", file=sys.stderr)
            return EXIT_BLOCKED

    documents = read_corpus(args.corpus)
    plan = build_sample_plan(represented_bytes=args.represented_bytes, protocol=protocol)
    tokenizer, plan, selections = build_fixture_tokenizer(documents, plan=plan, protocol=protocol)
    scope = SCOPE_FINAL if args.mode == "final" else SCOPE_FIXTURE

    if args.output_dir is not None:
        record = write_tokenizer_artifact(
            args.output_dir, tokenizer, plan, selections, build_scope=scope, protocol=protocol
        )
        print(json.dumps({key: record[key] for key in ("build_scope", "sample_plan_digest", "vocab_size")}, indent=2, sort_keys=True))

    print(f"build_scope: {scope}")
    print(f"sample_plan_digest: {plan.digest}")
    print("selected_bytes: " + json.dumps({s.source_id: s.selected_bytes for s in selections}, sort_keys=True))
    print("quota_reached: " + json.dumps({s.source_id: s.quota_reached for s in selections}, sort_keys=True))
    report = verify_tokenizer(tokenizer, protocol=protocol)
    print_report(report)
    return EXIT_OK if report.ok else EXIT_FAILED_VERIFICATION


if __name__ == "__main__":
    raise SystemExit(main())
