"""Show read-only progress for a restartable corpus pipeline state."""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = REPOSITORY_ROOT / "data" / "pipeline" / "slice_1pct.state.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh continuously at this interval; stop with Ctrl+C",
    )
    return parser.parse_args()


def scalar(connection: sqlite3.Connection, sql: str) -> int:
    return int(connection.execute(sql).fetchone()[0])


def snapshot(path: Path) -> str:
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
    dedup_percent = 100.0 if accepted == 0 else 100.0 * dedup / accepted
    if dedup < accepted:
        stage = "deduplication"
    elif decontamination < dedup:
        stage = "benchmark indexing/decontamination"
    elif assignments == 0:
        stage = "assignment/publication"
    else:
        stage = "assignment/publication or complete"
    return (
        f"{datetime.now().strftime('%H:%M:%S')} | stage: {stage} | "
        f"seen: {seen:,} | accepted: {accepted:,} | "
        f"dedup: {dedup:,}/{accepted:,} ({dedup_percent:.1f}%) | "
        f"decontaminated: {decontamination:,} | assigned: {assignments:,}"
    )


def main() -> int:
    args = parse_args()
    if args.watch is not None and args.watch <= 0:
        raise SystemExit("--watch interval must be positive")
    try:
        while True:
            print(snapshot(args.state), flush=True)
            if args.watch is None:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
