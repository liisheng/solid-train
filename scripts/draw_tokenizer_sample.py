"""Draw the deterministic 2 GiB source-stratified tokenizer sample (Plan Section 5.3).

Follows the frozen plan in configs/data/tokenizer_v1.yaml:
  quotas          from build_sample_plan (largest remainder on integer bytes)
  document_order  ascending sha256(salt | source_id | document_id) -- the frozen selection_key
  quota_fill_rule include documents in that order until target_bytes is reached

The frozen order is defined over "the source's documents". Enumerating every document in
FineWeb is not possible, so each source is drawn from a bounded CANDIDATE POOL: the first N
documents of the pinned revision's streaming order. The pool is deterministic given the
pinned revision, and its boundary is recorded in the manifest, so this interpretive choice is
written down rather than assumed.

Document ids must be unique or the selection key does not define an order. Where a source's
natural id repeats (OpenWebMath keys on url, and the same page appears more than once), the
collision is resolved with a content hash; byte-identical duplicates collapse and are dropped.
An assert enforces uniqueness before any selection happens.

Memory: only (document_id, utf8_bytes, file_offset) is held per pool document. Text is spooled
to disk and seek()-ed on the second pass, so a 1.5 GB selection costs kilobytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from datasets import load_dataset  # noqa: E402

from tinybench_lm.source_manifest import load_source_registry  # noqa: E402
from tinybench_lm.tokenizer import (  # noqa: E402
    build_sample_plan,
    load_tokenizer_protocol,
    selection_key,
)

POOL_FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
REPRESENTED = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = REPO / "data" / ("tokenizer_sample" if REPRESENTED is None else "tokenizer_sample_smoke")
OUT.mkdir(parents=True, exist_ok=True)

protocol = load_tokenizer_protocol()
registry = load_source_registry()
plan = (
    build_sample_plan(protocol=protocol)
    if REPRESENTED is None
    else build_sample_plan(protocol=protocol, represented_bytes=REPRESENTED)
)
sources = {s["source_id"]: s for s in registry["stable_sources"]}

print(f"plan digest      {plan.digest}")
print(f"sources digest   {plan.sources_digest}")
print(f"selection salt   {plan.selection_salt}")
print(f"represented      {plan.represented_bytes:,} bytes")
print(f"pool factor      {POOL_FACTOR}x quota")
print()

manifest: dict = {
    "scope": "FINAL_TOKENIZER_SAMPLE",
    "plan_digest": plan.digest,
    "sources_digest": plan.sources_digest,
    "selection_salt": plan.selection_salt,
    "represented_bytes": plan.represented_bytes,
    "pool_factor": POOL_FACTOR,
    "pool_definition": (
        "first N documents of the pinned revision's streaming order, N bounded by "
        "pool_factor x quota_bytes of accumulated UTF-8 text"
    ),
    "id_uniqueness": (
        "document ids are unique by construction; a repeated natural id is disambiguated "
        "with a 16-hex content hash, and byte-identical duplicates are dropped"
    ),
    "sources": [],
}

grand_selected = 0
started_all = time.perf_counter()


def resolve_id_field(row: dict) -> str:
    if isinstance(row.get("id"), str):
        return "id"
    if isinstance(row.get("url"), str):
        return "url"
    if isinstance(row.get("METADATA"), str):
        # gutenberg_english carries a stable Gutenberg text_id inside a JSON blob.
        return "METADATA.text_id"
    return "__index__"


def natural_id(row: dict, id_field: str, source_id: str, index: int) -> str:
    if id_field == "METADATA.text_id":
        try:
            return f"gutenberg:{json.loads(row['METADATA'])['text_id']}"
        except (ValueError, KeyError, TypeError):
            return f"{source_id}:{index:09d}"
    if id_field == "__index__":
        return f"{source_id}:{index:09d}"
    value = row.get(id_field)
    return str(value) if value else f"{source_id}:{index:09d}"


for quota in plan.quotas:
    spec = sources[quota.source_id]
    repo = spec["huggingface_repo"]
    revision = spec["intended_revision"]
    config = spec.get("huggingface_config")
    column = spec.get("text_column", "text")
    pool_target = int(quota.target_bytes * POOL_FACTOR)

    print(f"[{quota.source_id}] {repo} rev={revision[:12]} col={column}")
    print(f"    quota {quota.target_bytes:,}B   pool target {pool_target:,}B")

    kw = dict(split="train", streaming=True, revision=revision)
    if config:
        kw["name"] = config
    stream = load_dataset(repo, **kw)

    spool = OUT / f"{quota.source_id}.pool.jsonl"
    meta: list[tuple[str, int, int]] = []
    seen_ids: set[str] = set()
    collisions_resolved = 0
    duplicates_dropped = 0
    pool_bytes = 0
    id_field: str | None = None
    started = time.perf_counter()

    with spool.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(stream):
            text = row.get(column) or ""
            if not text.strip():
                continue
            if id_field is None:
                id_field = resolve_id_field(row)
            document_id = natural_id(row, id_field, quota.source_id, index)

            if document_id in seen_ids:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                candidate = f"{document_id}#{digest}"
                if candidate in seen_ids:
                    duplicates_dropped += 1
                    continue
                collisions_resolved += 1
                document_id = candidate
            seen_ids.add(document_id)

            size = len(text.encode("utf-8"))
            offset = handle.tell()
            handle.write(json.dumps({"document_id": document_id, "text": text}) + "\n")
            meta.append((document_id, size, offset))
            pool_bytes += size
            if pool_bytes >= pool_target:
                break

    assert len({item[0] for item in meta}) == len(meta), "document ids are not unique"
    elapsed = time.perf_counter() - started
    if collisions_resolved or duplicates_dropped:
        print(
            f"    id collisions resolved={collisions_resolved} "
            f"exact duplicates dropped={duplicates_dropped}"
        )
    print(
        f"    pool: {len(meta):,} docs, {pool_bytes:,}B in {elapsed:,.1f}s "
        f"({pool_bytes / 1e6 / max(elapsed, 1e-9):,.1f} MB/s), id_field={id_field}"
    )

    ordered = sorted(
        meta,
        key=lambda item: (selection_key(plan.selection_salt, quota.source_id, item[0]), item[0]),
    )
    chosen: list[tuple[str, int]] = []
    selected_bytes = 0
    for document_id, size, offset in ordered:
        if selected_bytes >= quota.target_bytes:
            break
        chosen.append((document_id, offset))
        selected_bytes += size

    out_path = OUT / f"{quota.source_id}.sample.jsonl"
    written = 0
    with spool.open("r", encoding="utf-8", newline="\n") as reader, out_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for document_id, offset in chosen:
            reader.seek(offset)
            payload = json.loads(reader.readline())
            assert payload["document_id"] == document_id, "offset does not match document"
            writer.write(
                json.dumps(
                    {
                        "source_id": quota.source_id,
                        "document_id": document_id,
                        "text": payload["text"],
                    }
                )
                + "\n"
            )
            written += 1

    for _ in range(10):
        try:
            spool.unlink()
            break
        except PermissionError:
            time.sleep(1.0)
    else:
        print(f"    note: could not remove {spool.name}; delete it manually")

    grand_selected += selected_bytes
    reached = selected_bytes >= quota.target_bytes
    print(f"    selected: {written:,} docs, {selected_bytes:,}B  quota_reached={reached}")
    print()

    manifest["sources"].append(
        {
            "source_id": quota.source_id,
            "huggingface_repo": repo,
            "revision": revision,
            "config": config,
            "text_column": column,
            "id_field": id_field,
            "target_bytes": quota.target_bytes,
            "pool_documents": len(meta),
            "pool_bytes": pool_bytes,
            "selected_documents": written,
            "selected_bytes": selected_bytes,
            "quota_reached": reached,
            "id_collisions_resolved": collisions_resolved,
            "exact_duplicates_dropped": duplicates_dropped,
            "stream_seconds": round(elapsed, 2),
            "output": out_path.name,
        }
    )

manifest["total_selected_bytes"] = grand_selected
manifest["total_seconds"] = round(time.perf_counter() - started_all, 2)
manifest["manifest_sha256"] = hashlib.sha256(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
(OUT / "sample_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print(
    f"TOTAL selected {grand_selected:,}B of {plan.represented_bytes:,}B target "
    f"({grand_selected / plan.represented_bytes:.1%}) in {manifest['total_seconds']:,.0f}s"
)
print(f"manifest -> {OUT / 'sample_manifest.json'}")
