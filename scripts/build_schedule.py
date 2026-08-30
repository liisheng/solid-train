"""Materialize and verify a deterministic mixture schedule (Plan Sections 5.4, 7.2, 8.3).

A mixture is an index schedule, not a repacked dataset. This script reads one split manifest
produced by ``scripts/build_shards.py``, materializes immutable
``(shard_id, token_offset, length)`` references with source tags, prints the schedule content
hash, and verifies the schedule against its manifest before anything trains on it.

The schedule content hash plus one integer ``schedule_cursor`` completely determine the
consumed training input, which is what makes a resume reproduce the exposure order exactly.
Nothing here downloads a corpus, produces shards, or starts training.

Usage::

    .venv\\Scripts\\python.exe scripts\\build_schedule.py \\
        --shard-root data\\shards \\
        --manifest data\\shards\\stable_train.manifest.json \\
        --output data\\schedules\\stable_train.schedule.json \\
        --sequence-length 1024 --seed 1337

Optional per-source sequence quotas materialize an exact mixture::

    --quota fineweb_edu=700 --quota dclm=200 --quota openwebmath=70 --quota narrative=30

Every source in the manifest must be given a quota when any quota is supplied; a partially
specified mixture fails closed rather than silently filling the remainder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinybench_lm.schedule import (
    available_sequences_per_source,
    build_materialized_schedule,
    format_schedule_report,
    load_schedule_protocol,
    verify_schedule,
    write_schedule,
)
from tinybench_lm.shards import load_split_manifest


def parse_quota(values: list[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    quotas: dict[str, int] = {}
    for item in values:
        source_id, separator, count = item.partition("=")
        if not separator or not count.strip().isdigit():
            raise ValueError(f"--quota expects source_id=sequences, got {item!r}")
        quotas[source_id.strip()] = int(count)
    return quotas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a deterministic index schedule over source-tagged shards",
    )
    parser.add_argument("--shard-root", type=Path, required=True, help="shard root the manifest indexes")
    parser.add_argument("--manifest", type=Path, required=True, help="split manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="schedule JSON to write")
    parser.add_argument("--sequence-length", type=int, required=True, help="loss tokens per sequence")
    parser.add_argument("--seed", type=int, required=True, help="ordering seed, recorded in the schedule")
    parser.add_argument(
        "--local-shuffle-buffer",
        type=int,
        default=None,
        help="bounded local shuffle window in sequences; defaults to the frozen contract value",
    )
    parser.add_argument(
        "--quota",
        action="append",
        metavar="SOURCE_ID=SEQUENCES",
        help="exact per-source sequence quota; repeat once per source",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_schedule_protocol()
    manifest = load_split_manifest(args.manifest)

    supply = available_sequences_per_source(manifest, sequence_length=args.sequence_length, protocol=protocol)
    print(f"split {manifest.split_id}: {len(manifest.shards)} shards, {manifest.token_count} tokens")
    for source_id, count in supply.items():
        print(f"  {source_id}: {count} sequences available")

    schedule = build_materialized_schedule(
        manifest,
        sequence_length=args.sequence_length,
        seed=args.seed,
        source_sequence_quotas=parse_quota(args.quota),
        local_shuffle_buffer_sequences=args.local_shuffle_buffer,
        protocol=protocol,
    )
    path = write_schedule(args.output, schedule)
    print()
    print(f"schedule_id:   {schedule.schedule_id}")
    print(f"content hash:  {schedule.content_hash()}")
    print(f"sequences:     {schedule.sequence_count} ({schedule.loss_tokens} loss tokens)")
    print(f"per source:    {schedule.sequences_per_source}")
    print(f"written to:    {path}")
    print()

    results = verify_schedule(manifest, schedule, protocol=protocol)
    print(format_schedule_report(results))
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
