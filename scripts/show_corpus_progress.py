"""Show read-only progress for a restartable corpus pipeline state."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = REPOSITORY_ROOT / "data" / "pipeline" / "slice_1pct.state.sqlite"
DEFAULT_BENCHMARK_INDEX = REPOSITORY_ROOT / "data" / "pipeline" / "benchmark_index.sqlite"
DEFAULT_BENCHMARK_EVIDENCE = (
    REPOSITORY_ROOT / "docs" / "evidence" / "decontamination" / "benchmark_inputs.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--benchmark-index", type=Path, default=DEFAULT_BENCHMARK_INDEX)
    parser.add_argument("--benchmark-evidence", type=Path, default=DEFAULT_BENCHMARK_EVIDENCE)
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh continuously at this interval; stop with Ctrl+C",
    )
    return parser.parse_args()


def scalar(connection: sqlite3.Connection, sql: str) -> int:
    return int(connection.execute(sql).fetchone()[0])


def benchmark_progress(index_path: Path, evidence_path: Path) -> tuple[int, int]:
    total = int(json.loads(evidence_path.read_text(encoding="utf-8"))["item_count"])
    resolved = index_path.resolve()
    if not resolved.is_file():
        return 0, total
    uri = f"file:{resolved.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2) as connection:
        exists = scalar(
            connection,
            "SELECT COUNT(1) FROM sqlite_master WHERE type = 'table' AND name = 'items'",
        )
        indexed = scalar(connection, "SELECT COUNT(1) FROM items") if exists else 0
    return indexed, total


def snapshot(path: Path, benchmark_index: Path, benchmark_evidence: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"pipeline state does not exist: {resolved}")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2) as connection:
        seen = scalar(connection, "SELECT COUNT(1) FROM documents")
        accepted = scalar(connection, "SELECT COUNT(1) FROM documents WHERE filter_action = 'KEEP'")
        dedup = scalar(connection, "SELECT COUNT(1) FROM dedup_decisions")
        decontamination = scalar(connection, "SELECT COUNT(1) FROM decontamination")
        assignments = scalar(connection, "SELECT COUNT(1) FROM assignments")
    indexed, benchmark_total = benchmark_progress(benchmark_index, benchmark_evidence)
    dedup_percent = 100.0 if accepted == 0 else 100.0 * dedup / accepted
    benchmark_percent = 100.0 if benchmark_total == 0 else 100.0 * indexed / benchmark_total
    if dedup < accepted:
        stage = "deduplication"
    elif indexed < benchmark_total:
        stage = "benchmark indexing"
    elif decontamination < dedup:
        stage = "decontamination"
    elif assignments == 0:
        stage = "assignment/publication"
    else:
        stage = "assignment/publication or complete"
    return (
        f"{datetime.now().strftime('%H:%M:%S')} | stage: {stage} | "
        f"seen: {seen:,} | accepted: {accepted:,} | "
        f"dedup: {dedup:,}/{accepted:,} ({dedup_percent:.1f}%) | "
        f"benchmark: {indexed:,}/{benchmark_total:,} ({benchmark_percent:.1f}%) | "
        f"decontaminated: {decontamination:,} | assigned: {assignments:,}"
    )


def main() -> int:
    args = parse_args()
    if args.watch is not None and args.watch <= 0:
        raise SystemExit("--watch interval must be positive")
    try:
        while True:
            print(
                snapshot(args.state, args.benchmark_index, args.benchmark_evidence),
                flush=True,
            )
            if args.watch is None:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
