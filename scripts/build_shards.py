"""Produce source-tagged uint16 shards and independent split manifests (Plan Sections 4.1-4.4, 5.4).

This is the final data output contract and it replaces the monolithic flat
``train.bin``/``validation.bin`` pair written by the pilot path
(``scripts/prepare_data.py``). Each source is packed into its own namespace, sources are
never pre-mixed on disk, and every split owns an independent manifest:

    stable/fineweb_edu/ ... reserved/math_prose/ ... validation_dev/... validation_final/...
    stable_train.manifest.json  reserved.manifest.json
    validation_dev.manifest.json  validation_final.manifest.json

The script reads documents from a **local** JSONL file. It never downloads a corpus and it
never produces billion-token shards; real-scale production stays operator-gated (see the
``readiness`` section of ``configs/data/shards_v1.yaml``).

Input JSONL, one object per line::

    {"document_id": "...", "source_id": "fineweb_edu", "text": "...",
     "boundary": "stable_train", "protected_slice": null}

``protected_slice`` is mandatory for ``validation_dev``/``validation_final`` documents.

Usage::

    .venv\\Scripts\\python.exe scripts\\build_shards.py --documents local.jsonl \\
        --tokenizer-dir artifacts\\tokenizer --output-dir data\\shards --scale FIXTURE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinybench_lm.shards import (
    RESERVED,
    SCALE_FINAL,
    SCALE_FIXTURE,
    STABLE_TRAIN,
    VALIDATION_DEV,
    VALIDATION_FINAL,
    ProfileDecisionRecord,
    ShardDocument,
    build_split_manifest,
    enforce_shard_isolation,
    format_shard_report,
    isolate_documents,
    load_shard_protocol,
    verify_mixture,
    verify_shard_files,
    write_split_manifest,
)
from tinybench_lm.source_manifest import (
    FINAL_TOKEN_COUNTER_ID,
    PROVISIONAL_TOKEN_COUNTER_ID,
    load_source_registry,
)
from tinybench_lm.tokenizer import load_tokenizer_artifact, load_tokenizer_protocol

SPLIT_ORDER = (RESERVED, STABLE_TRAIN, VALIDATION_DEV, VALIDATION_FINAL)


def read_documents(path: Path) -> list[ShardDocument]:
    """Read local shard candidates. Nothing is downloaded and no text is rewritten."""
    documents: list[ShardDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            missing = [name for name in ("document_id", "source_id", "text", "boundary") if not payload.get(name)]
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {missing}")
            documents.append(
                ShardDocument(
                    document_id=str(payload["document_id"]),
                    source_id=str(payload["source_id"]),
                    text=str(payload["text"]),
                    boundary=str(payload["boundary"]),
                    protected_slice=(str(payload["protected_slice"]) if payload.get("protected_slice") else None),
                )
            )
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack local documents into source-tagged uint16 shards with independent split manifests",
    )
    parser.add_argument("--documents", type=Path, required=True, help="local JSONL of shard candidates")
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="directory holding tokenizer.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="shard root")
    parser.add_argument(
        "--scale",
        choices=(SCALE_FIXTURE, SCALE_FINAL),
        default=SCALE_FIXTURE,
        help="FIXTURE defers measured token totals and shares; FINAL evaluates them",
    )
    parser.add_argument("--shard-document-budget", type=int, default=4096)
    parser.add_argument(
        "--token-counter-id",
        default=PROVISIONAL_TOKEN_COUNTER_ID,
        help=f"identity of the counter that produced the token totals; {FINAL_TOKEN_COUNTER_ID} for the final build",
    )
    parser.add_argument("--degraded-profile-id", default=None)
    parser.add_argument("--degraded-date", default=None, help="YYYY-MM-DD of the dated scope-reduction record")
    parser.add_argument("--degraded-owner", default=None)
    parser.add_argument("--degraded-reason", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_shard_protocol()
    registry = load_source_registry()
    tokenizer_protocol = load_tokenizer_protocol()
    tokenizer, _ = load_tokenizer_artifact(args.tokenizer_dir)

    documents = read_documents(args.documents)

    # Isolation runs across the complete candidate set, before anything is packed: a cluster
    # that crosses a boundary or a protected slice must fail closed, not land in a shard.
    isolation = isolate_documents(documents, protocol=protocol)
    enforce_shard_isolation(isolation)

    by_boundary: dict[str, list[ShardDocument]] = {}
    for document in documents:
        by_boundary.setdefault(document.boundary, []).append(document)

    manifests = {}
    results = []
    for split_id in SPLIT_ORDER:
        group = by_boundary.get(split_id)
        if not group:
            continue
        manifest = build_split_manifest(
            args.output_dir,
            tokenizer,
            group,
            split_id=split_id,
            shard_document_budget=args.shard_document_budget,
            protocol=protocol,
            registry=registry,
            tokenizer_protocol=tokenizer_protocol,
            token_counter_id=args.token_counter_id,
        )
        path = write_split_manifest(args.output_dir, manifest, protocol)
        manifests[split_id] = manifest
        results.extend(verify_shard_files(args.output_dir, manifest, protocol=protocol, registry=registry))
        print(f"{split_id}: {manifest.token_count} tokens, {len(manifest.shards)} shards -> {path}")

    decision_record = None
    if args.degraded_profile_id:
        decision_record = ProfileDecisionRecord(
            profile_id=args.degraded_profile_id,
            date=args.degraded_date or "",
            owner=args.degraded_owner or "",
            reason=args.degraded_reason or "",
        )

    results.extend(
        verify_mixture(
            manifests,
            scale=args.scale,
            decision_record=decision_record,
            isolation=isolation,
            protocol=protocol,
            registry=registry,
        )
    )
    print()
    print(format_shard_report(results))
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
