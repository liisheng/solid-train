"""Materialize the exact public benchmark text pinned for the production G1 scan.

The output is local, gitignored JSONL. A compact evidence manifest records every source
revision, split, row count, and the SHA-256 of the resulting benchmark-item stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.data_protocols import (  # noqa: E402
    PRODUCTION_DECONTAM_PROTOCOL_PATH,
    assert_ready_for_real_corpus_scan,
    load_decontamination_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "decontamination" / "benchmark_items.jsonl",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "evidence" / "decontamination" / "benchmark_inputs.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPOSITORY_ROOT / ".cache" / "huggingface")
    return parser.parse_args()


def _resolve_field(row: Mapping[str, Any], dotted_name: str) -> Any:
    value: Any = row
    for part in dotted_name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Mapping):
        found: list[str] = []
        for nested in value.values():
            found.extend(_strings(nested))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        found = []
        for nested in value:
            found.extend(_strings(nested))
        return found
    return []


def extract_texts(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    """Return ordered, non-empty, exact-deduplicated natural-language fields."""
    texts: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for text in _strings(_resolve_field(row, field)):
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return tuple(texts)


def _tasks(protocol: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    scope = protocol["benchmark_scope"]
    return list(scope["required_tasks"]) + list(scope["secondary_tasks"])


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = args.cache_dir.resolve()
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    from datasets import get_dataset_split_names, load_dataset

    protocol = load_decontamination_protocol(PRODUCTION_DECONTAM_PROTOCOL_PATH)
    assert_ready_for_real_corpus_scan(protocol)

    output = args.output.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.resolve().parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    total_items = 0
    task_records: list[dict[str, Any]] = []

    with partial.open("wb") as handle:
        for task in _tasks(protocol):
            task_id = str(task["task_id"])
            repo = str(task["dataset_repo"])
            revision = str(task["dataset_revision"])
            config = task.get("dataset_config")
            fields = tuple(str(field) for field in task["text_fields"])
            split_kwargs = {"path": repo, "revision": revision}
            if bool(task.get("trust_remote_code", False)):
                split_kwargs["trust_remote_code"] = True
            if config:
                split_kwargs["config_name"] = str(config)
            splits = sorted(get_dataset_split_names(**split_kwargs))

            task_count = 0
            split_counts: dict[str, int] = {}
            empty_rows_skipped: dict[str, int] = {}
            for split in splits:
                load_kwargs: dict[str, Any] = {
                    "path": repo,
                    "split": split,
                    "revision": revision,
                    "streaming": True,
                    "cache_dir": str(args.cache_dir.resolve()),
                    "trust_remote_code": bool(task.get("trust_remote_code", False)),
                }
                if config:
                    load_kwargs["name"] = str(config)
                dataset = load_dataset(**load_kwargs)
                split_count = 0
                skipped = 0
                for index, row in enumerate(dataset):
                    texts = extract_texts(row, fields)
                    if not texts:
                        skipped += 1
                        continue
                    payload = {
                        "task_id": task_id,
                        "item_id": f"{split}:{index}",
                        "split": split,
                        "texts": list(texts),
                    }
                    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                    handle.write(encoded)
                    digest.update(encoded)
                    split_count += 1
                split_counts[split] = split_count
                empty_rows_skipped[split] = skipped
                task_count += split_count
                print(f"{task_id}/{split}: {split_count:,} items ({skipped:,} empty rows skipped)", flush=True)
                if split_count == 0:
                    raise ValueError(f"{task_id}/{split} yielded no text from pinned fields {fields}")

            total_items += task_count
            task_records.append(
                {
                    "task_id": task_id,
                    "dataset_repo": repo,
                    "dataset_config": config,
                    "dataset_revision": revision,
                    "text_fields": list(fields),
                    "trust_remote_code": bool(task.get("trust_remote_code", False)),
                    "split_counts": split_counts,
                    "empty_rows_skipped": empty_rows_skipped,
                    "item_count": task_count,
                }
            )

    partial.replace(output)
    manifest = {
        "protocol": "production_decontamination_inputs",
        "protocol_version": "v1",
        "decontamination_protocol_digest": protocol["_digest"],
        "harness_commit": protocol["benchmark_scope"]["revision_pinning"]["harness_commit"],
        "benchmark_items_path": str(output.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "benchmark_items_sha256": digest.hexdigest(),
        "item_count": total_items,
        "tasks": task_records,
    }
    args.evidence.resolve().write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = materialize(args)
    print(f"items: {manifest['item_count']:,}")
    print(f"sha256: {manifest['benchmark_items_sha256']}")
    print(f"evidence: {args.evidence.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
