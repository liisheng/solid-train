"""Prepare or resume the isolated 5% reduced-scope corpus expansion.

The default invocation is read-only: it prints the pinned plan and does not create
state, contact a source, or publish an output.  ``--execute`` is deliberately
required to begin the expansion after an operator has reviewed the profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinybench_lm.benchmark_index import file_sha256  # noqa: E402
from tinybench_lm.corpus_pipeline import (  # noqa: E402
    load_acquisition_protocol,
    load_dedup_protocol,
    load_filter_protocol,
)
from tinybench_lm.data_protocols import (  # noqa: E402
    PRODUCTION_DECONTAM_PROTOCOL_PATH,
    load_decontamination_protocol,
)
from tinybench_lm.source_manifest import FINAL_TOKEN_COUNTER_ID, load_source_registry  # noqa: E402
from tinybench_lm.streaming_verify import verify_shard_outputs_streaming  # noqa: E402
from tinybench_lm.repeated_schedule import REPEAT_POLICY  # noqa: E402
from tinybench_lm.schedule import load_schedule, training_order_hash, verify_schedule  # noqa: E402
from tinybench_lm.shards import load_split_manifest  # noqa: E402
from tinybench_lm.training_recipe import load_training_recipe  # noqa: E402


RUN_ID = "reduced_5pct_v1"
TARGET_FRACTION = 0.05
MINIMUM_STABLE_TOKENS = 500_000_000
TARGET_STABLE_TOKENS = 550_000_000
SOURCE_STATE = ROOT / "data/pipeline/slice_1pct_v3_topup/state.sqlite"
STATE = ROOT / "data/pipeline/reduced_5pct_v1/state.sqlite"
OUTPUT_DIR = ROOT / "data/pipeline/reduced_5pct_v1_output"
RUN_DIR = ROOT / "runs/reduced_campaign/reduced_5pct_v1"
LOCK = ROOT / "runs/reduced_campaign/reduced_5pct_v1.lock"
PIPELINE_EVIDENCE = RUN_DIR / "pipeline.json"
SCOPE = ROOT / "configs/campaign/submission_scope_v1.yaml"
SCOPE_DIGEST = SCOPE.with_suffix(".yaml.sha256")
ACCEPTED = OUTPUT_DIR / "accepted.jsonl"
DECISIONS = OUTPUT_DIR / "decisions.jsonl"
BENCHMARK_INDEX = ROOT / "data/pipeline/benchmark_index.sqlite"
CACHE_DIR = ROOT / "data/hf_cache"

SOURCE_GROUPS = (
    ("base", 1.30, ("fineweb_edu", "dclm", "reserved_wikipedia")),
    ("math", 3.00, ("openwebmath",)),
    ("high_yield", 4.00, ("narrative", "reserved_textbook")),
)
SELECTION_TO_SOURCE = {
    "reserved_science": "reserved_science",
    "reserved_textbook": "reserved_textbook",
    "reserved_wikipedia": "reserved_wikipedia",
    "reserved_edu_decile": "reserved_edu_decile",
    "reserved_math_prose": "reserved_math_prose",
    "stable_train:fineweb_edu": "fineweb_edu",
    "stable_train:dclm": "dclm",
    "stable_train:openwebmath": "openwebmath",
    "stable_train:narrative": "narrative",
}
STABLE_SELECTIONS = tuple(key for key in SELECTION_TO_SOURCE if key.startswith("stable_train:"))


class ExpansionError(RuntimeError):
    """The expansion cannot safely proceed or satisfy reduced-scope checks."""


@dataclass(frozen=True)
class Check:
    check_id: str
    requirement: str
    observed: Any
    passed: bool

    def payload(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "requirement": self.requirement,
            "observed": self.observed,
            "status": "PASS" if self.passed else "FAIL",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="run the prepared acquisition/deduplication expansion")
    mode.add_argument("--verify-aggregate", action="store_true", help="verify completed reduced-scope shard aggregates")
    parser.add_argument("--resume", action="store_true", help="resume the exact journaled expansion after an interruption")
    parser.add_argument("--aggregate-output", type=Path, help="new JSON report path for --verify-aggregate")
    parser.add_argument("--shard-root", type=Path, help="completed shard root for --verify-aggregate")
    parser.add_argument(
        "--schedule",
        action="append",
        metavar="NAME=PATH",
        help="required schedule artifact to bind into --verify-aggregate; repeat for stable_train, validation_dev, recovery",
    )
    parser.add_argument("--profile-evidence", type=Path, help="new-manifest profile measurement.json required by --verify-aggregate")
    parser.add_argument("--recovery-evidence", type=Path, help="new-manifest recovery evidence.json required by --verify-aggregate")
    parser.add_argument("--pipeline-evidence", type=Path, default=PIPELINE_EVIDENCE)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return file_sha256(path)


@lru_cache(maxsize=1)
def _source_sha256() -> str:
    """Hash once per launcher process; active writers are refused before execution."""
    return _sha256(SOURCE_STATE)


def _scope_targets() -> dict[str, int]:
    registry = load_source_registry()
    acquisition = load_acquisition_protocol()
    stable = {
        f"stable_train:{item['source_id']}": int(int(item["target_tokens_at_11b"]) * TARGET_FRACTION)
        for item in registry["stable_sources"]
    }
    reserved = {
        str(item["source_id"]): int(int(item["target_tokens_at_minimum"]) * TARGET_FRACTION)
        for item in registry["reserved_sources"]
    }
    validation: dict[str, int] = {}
    for boundary in ("validation_dev", "validation_final"):
        total = int(int(acquisition["validation"]["targets"][boundary]) * TARGET_FRACTION)
        allocated = 0
        shares = list(acquisition["validation"]["stable_source_shares"].items())
        for index, (source_id, share) in enumerate(shares):
            quota = total - allocated if index == len(shares) - 1 else int(total * float(share) + 0.5)
            allocated += quota
            validation[f"{boundary}:{source_id}"] = quota
    return {**stable, **reserved, **validation}


def plan_payload() -> dict[str, Any]:
    registry = load_source_registry()
    acquisition = load_acquisition_protocol()
    physical = {
        source: _physical_target_tokens(source, registry, acquisition)
        for source in ("fineweb_edu", "dclm", "openwebmath", "narrative", "reserved_textbook", "reserved_wikipedia")
    }
    return {
        "run_id": RUN_ID,
        "mode": "PLAN_ONLY_NO_ACQUISITION",
        "target_fraction": TARGET_FRACTION,
        "stable_tokens": {"target": TARGET_STABLE_TOKENS, "minimum": MINIMUM_STABLE_TOKENS},
        "source_groups": [
            {
                "name": name,
                "pool_factor": factor,
                "sources": list(sources),
                "cumulative_pool_token_budgets": {
                    source: int(physical[source] * TARGET_FRACTION * factor) for source in sources
                },
            }
            for name, factor, sources in SOURCE_GROUPS
        ],
        "selection_token_targets": _scope_targets(),
        "source_state": str(SOURCE_STATE.relative_to(ROOT)),
        "new_state": str(STATE.relative_to(ROOT)),
        "new_output": str(OUTPUT_DIR.relative_to(ROOT)),
        "new_evidence": str(PIPELINE_EVIDENCE.relative_to(ROOT)),
        "safety": {
            "source_state_is_never_modified": True,
            "fresh_destinations_are_refused": True,
            "resume_requires_explicit_flag": True,
            "active_prepare_corpus_process_or_lock_refuses_execution": True,
            "publication_requires_complete_dedup_and_decontamination_coverage": True,
        },
    }


def _physical_target_tokens(source_id: str, registry: Mapping[str, Any], acquisition: Mapping[str, Any]) -> int:
    stable = {str(item["source_id"]): int(item["target_tokens_at_11b"]) for item in registry["stable_sources"]}
    reserved = {str(item["source_id"]): int(item["target_tokens_at_minimum"]) for item in registry["reserved_sources"]}
    validation_total = sum(int(value) for value in acquisition["validation"]["targets"].values())
    shares = acquisition["validation"]["stable_source_shares"]
    if source_id == "fineweb_edu":
        return stable[source_id] + reserved["reserved_science"] + reserved["reserved_edu_decile"] + round(validation_total * float(shares[source_id]))
    if source_id == "openwebmath":
        return stable[source_id] + reserved["reserved_math_prose"] + round(validation_total * float(shares[source_id]))
    if source_id in stable:
        return stable[source_id] + round(validation_total * float(shares[source_id]))
    return reserved[source_id]


def whole_document_share_tolerance(
    *, source_overshoot: int, target_share: float, total_overshoot: int, actual_total: int
) -> float:
    """A source's share allowance from observed, one-document rounding only."""
    if source_overshoot < 0 or total_overshoot < source_overshoot or actual_total <= 0:
        raise ValueError("invalid whole-document rounding inputs")
    return (source_overshoot + target_share * total_overshoot) / actual_total


def _active_conflicts() -> list[str]:
    """Return live corpus-launch commands without relying on optional psutil."""
    if os.name != "nt":
        return []
    command = "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise ExpansionError("unable to inspect active Windows processes; refusing expansion")
    try:
        rows = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise ExpansionError("active-process inspection returned invalid JSON; refusing expansion") from exc
    if isinstance(rows, Mapping):
        rows = [rows]
    by_pid = {int(row.get("ProcessId", -1)): row for row in rows}
    own_lineage = {os.getpid()}
    parent = by_pid.get(os.getpid(), {}).get("ParentProcessId")
    while parent is not None and int(parent) not in own_lineage:
        own_lineage.add(int(parent))
        parent = by_pid.get(int(parent), {}).get("ParentProcessId")
    matches = []
    for row in rows:
        image_name = str(row.get("Name") or "").casefold()
        command_line = str(row.get("CommandLine") or "")
        if "python" in image_name and ("prepare_corpus.py" in command_line or "expand_reduced_corpus.py" in command_line):
            if int(row.get("ProcessId", -1)) not in own_lineage:
                matches.append(f"PID {row.get('ProcessId')}: {command_line}")
    return matches


def _lock_payload() -> dict[str, Any]:
    return {"run_id": RUN_ID, "pid": os.getpid(), "source_state": str(SOURCE_STATE), "state": str(STATE)}


def _acquire_lock(*, resume: bool) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            current = json.loads(LOCK.read_text(encoding="utf-8"))
            owner = int(current.get("pid", -1))
            os.kill(owner, 0)
            raise ExpansionError(f"expansion lock is active for PID {owner}: {LOCK}")
        except ProcessLookupError:
            if not resume:
                raise ExpansionError(f"stale expansion lock exists; inspect it, then use --resume: {LOCK}")
            LOCK.unlink()
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            raise ExpansionError(f"cannot prove existing lock is stale: {LOCK}") from exc
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ExpansionError(f"expansion lock was acquired concurrently: {LOCK}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(_lock_payload(), handle, sort_keys=True)


def _release_lock() -> None:
    if LOCK.exists():
        try:
            if json.loads(LOCK.read_text(encoding="utf-8")).get("pid") == os.getpid():
                LOCK.unlink()
        except (OSError, json.JSONDecodeError):
            pass


def _clone_source_state(*, resume: bool) -> None:
    if STATE.exists():
        if not resume:
            raise ExpansionError(f"destination state already exists; use explicit --resume only after inspection: {STATE}")
        return
    if not SOURCE_STATE.is_file():
        raise ExpansionError(f"completed top-up state is absent: {SOURCE_STATE}")
    staging = STATE.with_suffix(".sqlite.staging")
    if staging.exists():
        raise ExpansionError(f"state staging path exists; inspect it before resuming: {staging}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{SOURCE_STATE.resolve().as_uri()}?mode=ro", uri=True)
    target = sqlite3.connect(staging)
    try:
        source.backup(target, pages=2048, sleep=0.01)
        if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ExpansionError("SQLite backup quick_check failed")
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.commit()
    finally:
        target.close()
        source.close()
    staging.replace(STATE)


def _journal_phase(connection: sqlite3.Connection) -> str | None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS expansion_v1_journal (run_id TEXT PRIMARY KEY, phase TEXT NOT NULL, source_sha256 TEXT NOT NULL)"
    )
    row = connection.execute("SELECT phase, source_sha256 FROM expansion_v1_journal WHERE run_id = ?", (RUN_ID,)).fetchone()
    if row is None:
        return None
    if str(row[1]) != _source_sha256():
        raise ExpansionError("source state changed since the expansion journal was created; refusing replay")
    return str(row[0])


def _prepare_state_for_dedup() -> None:
    with sqlite3.connect(STATE) as connection:
        phase = _journal_phase(connection)
        if phase is not None:
            return
        saved = "expansion_v1_saved_decontamination"
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (saved,)).fetchone()
        if exists:
            raise ExpansionError(f"unexpected saved-decontamination table exists: {saved}")
        original = int(connection.execute("SELECT COUNT(*) FROM decontamination").fetchone()[0])
        if not original:
            raise ExpansionError("source clone has no completed decontamination decisions to preserve")
        connection.execute(f"CREATE TABLE {saved} AS SELECT * FROM decontamination")
        copied = int(connection.execute(f"SELECT COUNT(*) FROM {saved}").fetchone()[0])
        if copied != original:
            raise ExpansionError("saved decontamination count does not match source clone")
        connection.execute("DELETE FROM decontamination")
        connection.execute("DELETE FROM assignments")
        connection.execute("DELETE FROM selection_keys")
        connection.execute(
            "INSERT INTO expansion_v1_journal(run_id, phase, source_sha256) VALUES (?, ?, ?)",
            (RUN_ID, "prepared", _source_sha256()),
        )


def _set_phase(expected: str, next_phase: str) -> None:
    with sqlite3.connect(STATE) as connection:
        phase = _journal_phase(connection)
        if phase == next_phase:
            return
        if phase != expected:
            raise ExpansionError(f"expected journal phase {expected!r}, found {phase!r}")
        connection.execute("UPDATE expansion_v1_journal SET phase = ? WHERE run_id = ?", (next_phase, RUN_ID))


def _restore_saved_decontamination() -> None:
    saved = "expansion_v1_saved_decontamination"
    with sqlite3.connect(STATE) as connection:
        phase = _journal_phase(connection)
        if phase == "restored":
            return
        if phase != "deduplicated":
            raise ExpansionError(f"cannot restore decontamination from journal phase {phase!r}")
        if connection.execute("SELECT COUNT(*) FROM decontamination").fetchone()[0]:
            raise ExpansionError("decontamination decisions appeared before controlled restoration")
        present = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (saved,)).fetchone()
        if not present:
            raise ExpansionError("saved decontamination table is absent")
        connection.execute(f"INSERT INTO decontamination SELECT * FROM {saved}")
        connection.execute(f"DROP TABLE {saved}")
        connection.execute("UPDATE expansion_v1_journal SET phase = ? WHERE run_id = ?", ("restored", RUN_ID))


def _base_command(*, stage: str, pool_factor: float, sources: tuple[str, ...], evidence: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/prepare_corpus.py"),
        "--state", str(STATE),
        "--cache-dir", str(CACHE_DIR),
        "--benchmark-index", str(BENCHMARK_INDEX),
        "--accepted-output", str(ACCEPTED),
        "--decisions-output", str(DECISIONS),
        "--evidence-output", str(evidence),
        "--target-fraction", str(TARGET_FRACTION),
        "--pool-factor", str(pool_factor),
        "--stage", stage,
    ]
    for source in sources:
        command.extend(("--source", source))
    return command


def _run(label: str, command: list[str]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = RUN_DIR / f"{label}.log"
    attempt = 1
    while log.exists():
        log = RUN_DIR / f"{label}.attempt-{attempt}.log"
        attempt += 1
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def _execute(*, resume: bool) -> None:
    conflicts = _active_conflicts()
    if conflicts:
        raise ExpansionError("conflicting corpus launcher is active: " + " | ".join(conflicts))
    _acquire_lock(resume=resume)
    try:
        _clone_source_state(resume=resume)
        _prepare_state_for_dedup()
        with sqlite3.connect(STATE) as connection:
            phase = _journal_phase(connection)
        if phase == "prepared":
            for label, factor, sources in SOURCE_GROUPS:
                _run(label, _base_command(stage="ingest", pool_factor=factor, sources=sources, evidence=RUN_DIR / f"{label}.json"))
            _set_phase("prepared", "ingested")
        with sqlite3.connect(STATE) as connection:
            phase = _journal_phase(connection)
        if phase == "ingested":
            _run("dedup", _base_command(stage="dedup", pool_factor=1.0, sources=(), evidence=RUN_DIR / "dedup.json"))
            _set_phase("ingested", "deduplicated")
        _restore_saved_decontamination()
        with sqlite3.connect(STATE) as connection:
            phase = _journal_phase(connection)
        if phase == "restored":
            # This source group is already at its factor-four cumulative budget, so the
            # all-stage pass performs no further ingestion while recording complete evidence.
            if PIPELINE_EVIDENCE.exists():
                _verify_pipeline_evidence(PIPELINE_EVIDENCE, require_files=True)
            else:
                if OUTPUT_DIR.exists():
                    raise ExpansionError("output bundle exists without final evidence; inspect rather than republishing")
                _run(
                    "pipeline",
                    _base_command(
                        stage="all",
                        pool_factor=4.0,
                        sources=("narrative", "reserved_textbook"),
                        evidence=PIPELINE_EVIDENCE,
                    ),
                )
            _verify_pipeline_evidence(PIPELINE_EVIDENCE, require_files=True)
            _set_phase("restored", "published")
    finally:
        _release_lock()


def _expected_protocol_digests() -> dict[str, str]:
    return {
        "acquisition": str(load_acquisition_protocol()["_digest"]),
        "sources": str(load_source_registry()["_digest"]),
        "decontamination": str(load_decontamination_protocol(PRODUCTION_DECONTAM_PROTOCOL_PATH)["_digest"]),
    }


def _scope_digest_is_pinned() -> bool:
    if not SCOPE.is_file() or not SCOPE_DIGEST.is_file():
        return False
    expected = SCOPE_DIGEST.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
    return expected == _sha256(SCOPE)


def _verify_pipeline_evidence(path: Path, *, require_files: bool) -> tuple[dict[str, Any], list[Check]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selections = {str(item["selection_id"]): item for item in payload.get("selection", [])}
    checks: list[Check] = [
        Check("reduced.pipeline_status", "completed 5% pipeline evidence is PASS", payload.get("status"), payload.get("status") == "PASS"),
        Check("reduced.target_fraction", "target fraction is 0.05", payload.get("target_fraction"), payload.get("target_fraction") == TARGET_FRACTION),
        Check("reduced.coverage", "deduplication and decontamination cover every filter-accepted document", payload.get("coverage"), payload.get("coverage", {}).get("status") == "PASS"),
        Check("reduced.scope_digest", "accepted frozen reduced-scope record matches its SHA-256 sidecar", str(SCOPE), _scope_digest_is_pinned()),
        Check("reduced.isolation", "zero boundary/slice violations across all declared boundaries and protected slices", payload.get("isolation"), payload.get("isolation", {}).get("status") == "PASS" and payload.get("isolation", {}).get("boundary_violations") == 0 and payload.get("isolation", {}).get("slice_violations") == 0 and payload.get("isolation", {}).get("undeclared_slices") == 0 and {"reserved", "stable_train", "validation_dev", "validation_final"}.issubset(set(payload.get("isolation", {}).get("boundaries", []))) and {"broad_general", "educational_science", "math_technical", "narrative_coreference"}.issubset(set(payload.get("isolation", {}).get("protected_slices", [])))),
        Check("reduced.protocol_digests", "pinned acquisition, sources, and v3 decontamination digests match", payload.get("protocol_digests"), payload.get("protocol_digests") == _expected_protocol_digests()),
    ]
    expected = _scope_targets()
    for selection_id, target in expected.items():
        observed = selections.get(selection_id, {})
        checks.append(Check(f"reduced.quota.{selection_id}", f"scaled {selection_id} target is {target}", observed.get("selected_tokens"), observed.get("status") == "PASS" and observed.get("target_tokens") == target and int(observed.get("selected_tokens", 0)) >= target))
    stable = sum(int(selections.get(key, {}).get("selected_tokens", 0)) for key in STABLE_SELECTIONS)
    checks.extend(
        [
            Check("reduced.stable_target", f"about {TARGET_STABLE_TOKENS} distinct stable tokens", stable, stable >= TARGET_STABLE_TOKENS),
            Check("reduced.stable_minimum", f"at least {MINIMUM_STABLE_TOKENS} distinct stable tokens", stable, stable >= MINIMUM_STABLE_TOKENS),
        ]
    )
    if require_files:
        for label, output, digest in (("accepted", payload.get("accepted_output"), payload.get("accepted_sha256")), ("decisions", payload.get("decisions_output"), payload.get("decisions_sha256")), ("state", payload.get("sqlite_state"), payload.get("sqlite_state_sha256"))):
            candidate = Path(str(output)) if output else Path()
            candidate = candidate if candidate.is_absolute() else ROOT / candidate
            checks.append(Check(f"reduced.hash.{label}", f"{label} artifact exists and matches recorded SHA-256", str(candidate), candidate.is_file() and bool(digest) and _sha256(candidate) == digest))
    return payload, checks


def _verify_state_bindings(state_path: Path) -> list[Check]:
    expected = {
        "acquisition_digest": load_acquisition_protocol()["_digest"],
        "sources_digest": load_source_registry()["_digest"],
        "filters_digest": load_filter_protocol()["_digest"],
        "dedup_digest": load_dedup_protocol()["_digest"],
        "token_counter_id": FINAL_TOKEN_COUNTER_ID,
    }
    with sqlite3.connect(f"{state_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        decontamination = int(connection.execute("SELECT COUNT(*) FROM decontamination").fetchone()[0])
        dedup = int(connection.execute("SELECT COUNT(*) FROM dedup_decisions").fetchone()[0])
    return [
        Check("reduced.state_contract", "state binds pinned acquisition/source/filter/dedup/tokenizer contracts", {key: metadata.get(key) for key in expected}, all(metadata.get(key) == value for key, value in expected.items())),
        Check("reduced.state_decision_coverage", "state contains decontamination decisions for every dedup decision", {"dedup": dedup, "decontamination": decontamination}, dedup == decontamination),
    ]


def _parse_named_paths(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values or []:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ExpansionError(f"--schedule expects NAME=PATH, got {item!r}")
        if name in result:
            raise ExpansionError(f"duplicate --schedule name: {name}")
        candidate = Path(raw_path)
        result[name] = candidate if candidate.is_absolute() else ROOT / candidate
    return result


def _schedule_checks(paths: Mapping[str, Path], shard_root: Path) -> tuple[list[Check], dict[str, str], dict[str, str]]:
    required = {"stable_train", "validation_dev", "recovery"}
    checks = [Check("reduced.schedule_set", "new expansion has stable_train, validation_dev, and recovery schedules", sorted(paths), set(paths) == required)]
    digests: dict[str, str] = {}
    content_hashes: dict[str, str] = {}
    manifest_names = {"stable_train": "stable_train.manifest.json", "validation_dev": "validation_dev.manifest.json", "recovery": "stable_train.manifest.json"}
    for name, path in paths.items():
        present = path.is_file()
        digest = _sha256(path) if present else "ABSENT"
        digests[name] = digest
        checks.append(Check(f"reduced.schedule.{name}", "schedule artifact is present and SHA-256 recorded", str(path), present))
        if not present or name not in manifest_names:
            continue
        manifest_path = shard_root / manifest_names[name]
        try:
            manifest = load_split_manifest(manifest_path)
            schedule = load_schedule(path)
            results = verify_schedule(manifest, schedule)
            failures = [result for result in results if result.failed]
            content_hashes[name] = schedule.content_hash()
            checks.append(Check(f"reduced.schedule_semantics.{name}", "manifest/protocol/references/quotas/determinism/cursor all pass", [result.check_id for result in failures], not failures))
        except Exception as exc:  # verifier exceptions are fail-closed evidence failures
            checks.append(Check(f"reduced.schedule_semantics.{name}", "schedule verifier can load and validate the expansion schedule", str(exc), False))
    return checks, digests, content_hashes


def _runtime_schedule_identity(base_hash: str, epochs: int) -> str:
    if epochs < 1:
        raise ValueError("train_epochs must be positive")
    if epochs == 1:
        return str(base_hash)
    return hashlib.sha256(json.dumps({
        "policy": REPEAT_POLICY,
        "base_schedule_hash": str(base_hash),
        "epochs": int(epochs),
    }, sort_keys=True).encode()).hexdigest()


def _profile_update_hashes_match(
    metric_rows: list[Mapping[str, Any]], *, schedule: Any, epochs: int,
    sequences_per_update: int,
) -> bool:
    if epochs < 1 or sequences_per_update < 1:
        return False
    repeated_entries = tuple(schedule.entries) * int(epochs)
    for index, row in enumerate(metric_rows):
        cursor = (index + 1) * int(sequences_per_update)
        if cursor > len(repeated_entries) or row.get("schedule_cursor") != cursor:
            return False
        expected = training_order_hash(repeated_entries[cursor - sequences_per_update : cursor])
        if row.get("train_batch_reference_hash") != expected:
            return False
    return bool(metric_rows)


def _profile_checks(
    path: Path | None, *, shard_root: Path, schedule_hashes: Mapping[str, str]
) -> tuple[list[Check], dict[str, Any] | None]:
    if path is None:
        return [Check("reduced.profile", "new-manifest profile measurement is supplied", "absent", False)], None
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.is_file():
        return [Check("reduced.profile", "new-manifest profile measurement is supplied", str(resolved), False)], None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    measured = float(payload.get("window_seconds", 0))
    required_metrics = {"throughput_p10", "throughput_median", "throughput_p90", "weighted_tokens_per_second", "wall_seconds", "optimizer_seconds", "peak_allocated_vram_gib"}
    profile_dir = resolved.parent
    run_config_path = profile_dir / "train/run_config.json"
    identity_path = profile_dir / "train/run_identity.json"
    metrics_path = profile_dir / "train/metrics.jsonl"
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        metric_rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_schedule = load_schedule(run_config["training_args"]["train_schedule"])
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return [Check("reduced.profile_inputs", "profile run configuration, identity, and update metrics exist", str(exc), False)], payload
    metadata = run_config.get("data_metadata", {})
    training_args = run_config.get("training_args", {})
    batch_plan = run_config.get("batch_plan", {})
    epochs = int(metadata.get("train_epochs", 0))
    sequences_per_update = int(batch_plan.get("sequences_per_update", 0))
    runtime_schedule_hash = _runtime_schedule_identity(schedule_hashes.get("stable_train", ""), epochs) if epochs >= 1 else ""
    semantic = identity.get("semantics", {})
    recipe = load_training_recipe()
    def same_path(value, expected):
        return (ROOT / str(value)).resolve() == expected.resolve()
    inputs_ok = (
        same_path(training_args.get("shard_root", ""), shard_root)
        and same_path(training_args.get("train_manifest", ""), shard_root / "stable_train.manifest.json")
        and same_path(training_args.get("validation_manifest", ""), shard_root / "validation_dev.manifest.json")
        and metadata.get("train_schedule_content_hash") == runtime_schedule_hash
        and metadata.get("validation_schedule_content_hash") == schedule_hashes.get("validation_dev")
        and run_config.get("recipe_digest") == recipe["_digest"]
        and identity.get("run_id") == run_config.get("run_id")
        and semantic.get("recipe_digest") == recipe["_digest"]
        and semantic.get("train_schedule_content_hash") == runtime_schedule_hash
        and sequences_per_update == 256
        and int(batch_plan.get("loss_tokens_per_update", 0)) == 262144
        and int(batch_plan.get("sequence_length", 0)) == 1024
        and all(row.get("schedule_content_hash") == runtime_schedule_hash for row in metric_rows)
        and _profile_update_hashes_match(metric_rows, schedule=train_schedule, epochs=epochs, sequences_per_update=sequences_per_update)
    )
    return [
        Check("reduced.profile", "profile has >=1800 measured optimizer seconds after startup exclusion", {"status": payload.get("status"), "measured_seconds": measured}, payload.get("status") == "MEASURED" and measured >= 1800),
        Check("reduced.profile_metrics", "profile records p10/median/p90, weighted rate, wall/optimizer time, and peak memory", sorted(required_metrics - set(payload)), required_metrics.issubset(payload) and payload.get("used_real_shards") is True),
        Check("reduced.profile_inputs", "profile recipe, manifests, schedules, and per-update input hashes bind to the expansion bundle", {"run_id": run_config.get("run_id"), "metric_rows": len(metric_rows)}, inputs_ok),
    ], payload


RECOVERY_EXACT_RESUME_FIELDS = frozenset({
    "model", "optimizer", "scaler", "rng_state", "data_rng_state", "counters",
    "run_id", "best_validation_state", "schedule_cursor", "schedule_content_hash",
})


def _recovery_configs_bind_inputs(
    run_configs: list[Mapping[str, Any]], *, shard_root: Path,
    schedule_hashes: Mapping[str, str], recipe_digest: str,
) -> bool:
    def same_path(value, expected_path):
        return (ROOT / str(value)).resolve() == expected_path.resolve()
    try:
        for config in run_configs:
            metadata = config.get("data_metadata", {})
            epochs = int(metadata.get("train_epochs", 0))
            if not (
                same_path(config.get("training_args", {}).get("shard_root", ""), shard_root)
                and same_path(config.get("training_args", {}).get("train_manifest", ""), shard_root / "stable_train.manifest.json")
                and same_path(config.get("training_args", {}).get("validation_manifest", ""), shard_root / "validation_dev.manifest.json")
                and metadata.get("train_schedule_content_hash") == _runtime_schedule_identity(schedule_hashes.get("recovery", ""), epochs)
                and metadata.get("validation_schedule_content_hash") == schedule_hashes.get("validation_dev")
                and config.get("recipe_digest") == recipe_digest
            ):
                return False
        return True
    except (TypeError, ValueError):
        return False


def _recovery_checks(
    path: Path | None, *, shard_root: Path, schedule_hashes: Mapping[str, str]
) -> tuple[list[Check], dict[str, Any] | None]:
    if path is None:
        return [Check("reduced.recovery", "fresh expansion recovery evidence is supplied", "absent", False)], None
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.is_file():
        return [Check("reduced.recovery", "fresh expansion recovery evidence is supplied", str(resolved), False)], None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    identity = payload.get("training_input_identity", {})
    exact = set(payload.get("exact_resume_fields", []))
    expected = RECOVERY_EXACT_RESUME_FIELDS
    try:
        run_configs = [
            json.loads((resolved.parent / name / "run_config.json").read_text(encoding="utf-8"))
            for name in ("uninterrupted", "resumed")
        ]
        recipe_digest = load_training_recipe()["_digest"]
        inputs_ok = _recovery_configs_bind_inputs(
            run_configs, shard_root=shard_root, schedule_hashes=schedule_hashes, recipe_digest=recipe_digest
        )
    except (OSError, json.JSONDecodeError) as exc:
        inputs_ok = False
        run_configs = []
        recovery_config_error = str(exc)
    else:
        recovery_config_error = ""
    valid = (
        payload.get("status") == "LOCAL_CHECKS_PASS"
        and payload.get("corruption_rejected") is True
        and identity.get("base_schedule_content_hash") == schedule_hashes.get("recovery")
        and expected.issubset(exact)
        and bool(identity.get("batch_reference_hashes"))
        and payload.get("export", {}).get("ok") is True
    )
    return [
        Check("reduced.recovery", "fresh recovery/resume/corruption/export evidence binds to the validated expansion recovery schedule", {"status": payload.get("status"), "schedule": identity.get("base_schedule_content_hash")}, valid),
        Check("reduced.recovery_inputs", "both recovery run configurations bind expansion shard root/manifests, recovery schedule, and recipe", {"run_configs": len(run_configs), "error": recovery_config_error}, inputs_ok),
    ], payload


def _verify_aggregate(
    shard_root: Path,
    evidence_path: Path,
    output: Path | None,
    schedules: Mapping[str, Path],
    profile_evidence: Path | None,
    recovery_evidence: Path | None,
) -> dict[str, Any]:
    evidence, checks = _verify_pipeline_evidence(evidence_path, require_files=True)
    state_value = Path(str(evidence["sqlite_state"]))
    checks.extend(_verify_state_bindings(state_value if state_value.is_absolute() else ROOT / state_value))
    schedule_checks, schedule_digests, schedule_hashes = _schedule_checks(schedules, shard_root)
    checks.extend(schedule_checks)
    profile_checks, profile = _profile_checks(profile_evidence, shard_root=shard_root, schedule_hashes=schedule_hashes)
    checks.extend(profile_checks)
    recovery_checks, recovery = _recovery_checks(recovery_evidence, shard_root=shard_root, schedule_hashes=schedule_hashes)
    checks.extend(recovery_checks)
    manifest_names = ("stable_train.manifest.json", "validation_dev.manifest.json", "validation_final.manifest.json", "reserved.manifest.json")
    checks.append(Check("reduced.manifest_set", "all four new expansion split manifests are present", list(manifest_names), all((shard_root / name).is_file() for name in manifest_names)))
    report = verify_shard_outputs_streaming(shard_root, scale="FIXTURE", isolation_evidence=evidence_path)
    integrity_failures = [item for item in report.results if item.status == "FAIL"]
    checks.append(Check("reduced.shard_integrity", "bounded streaming shard verification has no integrity failure", len(integrity_failures), not integrity_failures))
    sources = report.facts.get("source_tokens", {})
    selections = {str(item["selection_id"]): item for item in evidence["selection"]}
    for selection_id, source_id in SELECTION_TO_SOURCE.items():
        boundary = "stable_train" if selection_id.startswith("stable_train:") else "reserved"
        observed = int(sources.get(boundary, {}).get(source_id, 0))
        required = int(selections[selection_id]["selected_tokens"])
        checks.append(Check(f"reduced.shard_tokens.{selection_id}", "packed split/source tokens cover the selected corpus tokens", observed, observed >= required))
    for boundary in ("validation_dev", "validation_final"):
        for source_id in ("fineweb_edu", "dclm", "openwebmath", "narrative"):
            selection_id = f"{boundary}:{source_id}"
            observed = int(sources.get(boundary, {}).get(source_id, 0))
            required = int(selections[selection_id]["selected_tokens"])
            checks.append(Check(f"reduced.shard_tokens.{selection_id}", "packed split/source tokens cover the selected corpus tokens", observed, observed >= required))
    with sqlite3.connect(f"{(ROOT / Path(str(evidence['sqlite_state']))).resolve().as_uri()}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT source_id, SUM(d.token_count), COUNT(*), MAX(d.token_count) FROM assignments a JOIN documents d USING(doc_key) WHERE boundary = 'stable_train' GROUP BY source_id").fetchall()
    selected = {str(source): {"tokens": int(tokens), "documents": int(documents), "maximum": int(maximum)} for source, tokens, documents, maximum in rows}
    stable_total = sum(value["tokens"] for value in selected.values())
    scope_shares = {"fineweb_edu": 0.70, "dclm": 0.20, "openwebmath": 0.07, "narrative": 0.03}
    targets = _scope_targets()
    total_overshoot = sum(selected.get(source, {}).get("tokens", 0) - int(targets[f"stable_train:{source}"]) for source in scope_shares)
    for source, target_share in scope_shares.items():
        entry = selected.get(source, {})
        selected_tokens = entry.get("tokens", 0)
        target_tokens = int(targets[f"stable_train:{source}"])
        overshoot = selected_tokens - target_tokens
        maximum = entry.get("maximum", 0)
        checks.append(Check(f"reduced.rounding.{source}", "each source exceeds its quota by no more than its own final whole document", {"overshoot": overshoot, "maximum_document_tokens": maximum}, 0 <= overshoot <= maximum))
        actual_share = selected_tokens / stable_total if stable_total else 0.0
        tolerance = whole_document_share_tolerance(source_overshoot=overshoot, target_share=target_share, total_overshoot=total_overshoot, actual_total=stable_total) if stable_total else 1.0
        checks.append(Check(f"reduced.actual_share.{source}", f"actual stable share {target_share:.0%} within observed whole-document rounding envelope", {"actual": actual_share, "target": target_share, "source_overshoot": overshoot, "total_overshoot": total_overshoot, "tolerance": tolerance}, abs(actual_share - target_share) <= tolerance + 1e-12))
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    payload = {
        "status": status,
        "scope": "REDUCED_SCOPE_ONLY",
        "does_not_claim": [
            "original 11B-token G1 corpus threshold",
            "frozen full-scale shard profile selection or full validation-size thresholds",
            "original multi-experiment G3 campaign",
            "other-machine verification",
            "baseline training",
            "fresh full evaluation",
            "public assets, compute disclosure, or video/screenshots",
            "human release approval",
        ],
        "checks": [check.payload() for check in checks],
        "manifest_sha256": {
            name: _sha256(shard_root / name)
            for name in manifest_names
            if (shard_root / name).is_file()
        },
        "schedule_sha256": schedule_digests,
        "profile_measurement": profile,
        "recovery_evidence": recovery,
        "streaming_integrity_result": report.to_dict(),
    }
    if output:
        output = output if output.is_absolute() else ROOT / output
        if output.exists():
            raise ExpansionError(f"refusing to overwrite aggregate report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    if args.verify_aggregate:
        if args.resume or not args.shard_root:
            raise SystemExit("--verify-aggregate requires --shard-root and cannot use --resume")
        result = _verify_aggregate(
            args.shard_root,
            args.pipeline_evidence,
            args.aggregate_output,
            _parse_named_paths(args.schedule),
            args.profile_evidence,
            args.recovery_evidence,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    if args.execute:
        _execute(resume=args.resume)
        return 0
    if args.resume:
        raise SystemExit("--resume is only valid with --execute")
    print(json.dumps(plan_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExpansionError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"expansion refused: {exc}") from exc
