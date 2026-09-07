"""Preserve an offline v2 corpus state and publish a separate v3 reclassification state."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.benchmark_index import file_sha256  # noqa: E402
from tinybench_lm.corpus_pipeline import load_acquisition_protocol  # noqa: E402
from tinybench_lm.data_protocols import (  # noqa: E402
    FROZEN_PROTOCOL_SHA256,
    PRODUCTION_DECONTAM_PROTOCOL_PATH,
    load_decontamination_protocol,
)


def fork_state(source: Path, destination: Path, *, legacy_protocol_digest: str) -> dict:
    """SQLite backup includes committed WAL pages; only the new copy is reset.

    Stop the source writer first. Legacy states did not bind decisions to a
    protocol, so the operator must explicitly attest the historical v2 digest.
    Unpublished staging directories remain inspectable after an error.
    """
    source, destination = source.resolve(), destination.resolve()
    protocol = load_decontamination_protocol(PRODUCTION_DECONTAM_PROTOCOL_PATH)
    benchmark_hash = str(load_acquisition_protocol()["decontamination"]["benchmark_items_sha256"])
    if legacy_protocol_digest != FROZEN_PROTOCOL_SHA256["decontam_v2.yaml"]:
        raise ValueError("only the frozen v2 state may be migrated by this tool")
    if not source.is_file():
        raise ValueError(f"source state is absent: {source}")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    staging = destination.with_name(f".{destination.name}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    target = staging / "state.sqlite"
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        # This snapshot is a consistent copy even when the source still has a WAL.
        with closing(sqlite3.connect(target)) as copied:
            original.backup(copied)
    copied = sqlite3.connect(target)
    try:
        if copied.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("copied state failed SQLite quick_check")
        metadata = dict(copied.execute("SELECT key, value FROM metadata"))
        if metadata.get("decontamination_protocol_digest", legacy_protocol_digest) != legacy_protocol_digest:
            raise ValueError("source decision protocol is not v2")
        if metadata.get("decontamination_benchmark_sha256", benchmark_hash) != benchmark_hash:
            raise ValueError("source benchmark hash mismatch")
        for table in ("assignments", "selection_keys"):
            if copied.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
                raise ValueError("downstream selection already exists; refusing to invalidate it")
        preserved = {
            table: copied.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("documents", "source_cursors", "ingest_events", "dedup_decisions",
                          "representatives", "minhash_bands", "boundary_review_pairs")
        }
        counts = copied.execute(
            "SELECT action, reason_code, COUNT(*) FROM decontamination GROUP BY action, reason_code"
        ).fetchall()
        copied.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        snapshot_sha256 = file_sha256(target)
        report = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "source_state": str(source),
            "destination_state": str(destination / "state.sqlite"),
            "source_snapshot_sha256": snapshot_sha256,
            "previous_protocol_digest": legacy_protocol_digest,
            "previous_binding": "recorded" if "decontamination_protocol_digest" in metadata else "operator_attested_legacy_v2",
            "new_protocol_digest": protocol["_digest"],
            "benchmark_sha256": benchmark_hash,
            "previous_decision_counts": counts,
            "preserved_table_counts": preserved,
            "reason": "v2 standalone short benchmark answers quarantined unrelated corpus documents",
        }
        with copied:
            copied.execute("DELETE FROM decontamination")
            copied.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (
                ("decontamination_protocol_digest", protocol["_digest"]),
                ("decontamination_benchmark_sha256", benchmark_hash),
                ("decontamination_migration", json.dumps(report, sort_keys=True)),
            ))
        if copied.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
            raise ValueError("new state WAL checkpoint was blocked")
    finally:
        copied.close()
    report["initial_destination_sha256"] = file_sha256(target)
    (staging / "migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True, help="new directory for state.sqlite and migration.json")
    parser.add_argument("--legacy-protocol-digest", required=True, help="explicit attestation for unbound legacy v2 decisions")
    args = parser.parse_args()
    print(json.dumps(fork_state(args.source, args.destination, legacy_protocol_digest=args.legacy_protocol_digest), indent=2))


if __name__ == "__main__":
    main()
