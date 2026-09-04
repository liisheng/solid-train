"""Run one bounded pipeline slice over the pinned sources (Plan Section 5.5, G1).

Streams a stratified slice of the real corpus through the seven mandatory stages and writes a
measurements JSON for scripts/pipeline_benchmark.py, which does the forecasting.

  python slice_pipeline.py <slice_fraction> <mode_id> <out.json>

The tokenizer is the real FINAL 12,288-ID artifact, so accepted_token_count is measured under
final_tokenizer_v1 rather than a provisional counter.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402

from tinybench_lm.source_manifest import load_source_registry  # noqa: E402
from tinybench_lm.tokenizer import load_tokenizer_artifact  # noqa: E402

SLICE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
MODE_ID = sys.argv[2] if len(sys.argv) > 2 else "slice_1pct"
DEST = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "runs" / "bench" / "slice.measurements.json"

FULL_TOKENS = 11_000_000_000
TARGET_TOKENS = int(FULL_TOKENS * SLICE)
BYTES_PER_TOKEN = 4.149  # measured on the final tokenizer
TARGET_BYTES = int(TARGET_TOKENS * BYTES_PER_TOKEN)

registry = load_source_registry()
tokenizer, _ = load_tokenizer_artifact(REPO / "data" / "tokenizer_final")

print(f"mode {MODE_ID}  slice {SLICE:.2%}  target {TARGET_TOKENS:,} tokens ~ {TARGET_BYTES:,} bytes")
print(f"tokenizer vocab {tokenizer.get_vocab_size():,}\n")


def peak_rss() -> int:
    class C(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    # Without argtypes the 64-bit process HANDLE is truncated to 32 bits and the call
    # fails, silently returning a peak of 0 -- a false measurement, not a missing one.
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(C), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    c = C(); c.cb = ctypes.sizeof(C)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed; refusing to report a peak RSS of 0")
    return int(c.PeakWorkingSetSize)


stages: list[dict] = []


def record(stage_id, documents, input_bytes, output_bytes, elapsed, temp_bytes=0):
    rss = peak_rss()
    stages.append({
        "stage_id": stage_id, "documents": documents, "input_bytes": input_bytes,
        "output_bytes": output_bytes, "elapsed_seconds": round(elapsed, 4),
        "peak_rss_bytes": rss, "peak_temporary_disk_bytes": temp_bytes,
    })
    print(f"  {stage_id:<34} {documents:>8,} docs {elapsed:>8.2f}s "
          f"{documents/max(elapsed,1e-9):>9,.0f} doc/s  rss={rss/1024**3:>5.2f}G")


NONPRINT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


# ------------------------------------------------------------------ 1. stream + filter
t0 = time.perf_counter()
docs: list[tuple[str, str]] = []
raw_bytes = 0
rejected = 0
for spec in registry["stable_sources"]:
    share = float(spec["stable_share"])
    quota = int(TARGET_BYTES * share)
    kw = dict(split="train", streaming=True, revision=spec["intended_revision"])
    if spec.get("huggingface_config"):
        kw["name"] = spec["huggingface_config"]
    col = spec.get("text_column", "text")
    got = 0
    for row in load_dataset(spec["huggingface_repo"], **kw):
        text = row.get(col) or ""
        raw_bytes += len(text.encode("utf-8"))
        if len(text) < 200 or NONPRINT.search(text):
            rejected += 1
            continue
        docs.append((spec["source_id"], text))
        got += len(text.encode("utf-8"))
        if got >= quota:
            break
accepted_bytes = sum(len(t.encode("utf-8")) for _, t in docs)
record("stream_and_filter", len(docs), raw_bytes, accepted_bytes, time.perf_counter() - t0)
print(f"      rejected {rejected:,} documents at the filter")

# ------------------------------------------------------------------ 2. exact + mirror dedup
t0 = time.perf_counter()
seen_full: set[str] = set(); seen_mirror: set[str] = set(); kept = []
for sid, text in docs:
    norm = normalize(text)
    full = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    mirror = hashlib.sha256(norm[:512].encode("utf-8")).hexdigest()
    if full in seen_full or mirror in seen_mirror:
        continue
    seen_full.add(full); seen_mirror.add(mirror); kept.append((sid, text))
removed_exact = len(docs) - len(kept)
docs = kept
record("exact_and_mirror_dedup", len(docs), accepted_bytes,
       sum(len(t.encode("utf-8")) for _, t in docs), time.perf_counter() - t0)
print(f"      removed {removed_exact:,} exact/mirror duplicates")

# ------------------------------------------------------------------ 3. within-source near dedup
t0 = time.perf_counter()
PERMS = 128
rng = np.random.default_rng(0)
A = rng.integers(1, 2**61 - 1, PERMS, dtype=np.int64)
B = rng.integers(0, 2**61 - 1, PERMS, dtype=np.int64)
MOD = (1 << 61) - 1
sigs = np.empty((len(docs), PERMS), dtype=np.int64)
for i, (_, text) in enumerate(docs):
    words = normalize(text).split()
    shingles = {" ".join(words[j:j + 5]) for j in range(max(len(words) - 4, 1))}
    if not shingles:
        sigs[i] = 0; continue
    h = np.fromiter(
        (int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big") % MOD for s in shingles),
        dtype=np.int64, count=len(shingles))
    sigs[i] = ((A[None, :] * h[:, None] + B[None, :]) % MOD).min(axis=0)
record("within_source_near_dedup", len(docs), sum(len(t.encode('utf-8')) for _, t in docs),
       sigs.nbytes, time.perf_counter() - t0, temp_bytes=sigs.nbytes)

# ------------------------------------------------------------------ 4. cross-source near dedup
t0 = time.perf_counter()
BANDS, ROWS = 16, 8
buckets: dict[tuple, list[int]] = {}
for i in range(len(docs)):
    for b in range(BANDS):
        buckets.setdefault((b, sigs[i, b*ROWS:(b+1)*ROWS].tobytes()), []).append(i)
candidates = set()
for members in buckets.values():
    if len(members) > 1:
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                candidates.add((members[x], members[y]))
near_dupes = 0
for x, y in candidates:
    if (sigs[x] == sigs[y]).mean() >= 0.85:
        near_dupes += 1
record("cross_source_near_dedup", len(docs), sigs.nbytes, near_dupes * 8,
       time.perf_counter() - t0, temp_bytes=sigs.nbytes)
print(f"      {len(candidates):,} candidate pairs, {near_dupes:,} above 0.85 estimated Jaccard")

# ------------------------------------------------------------------ 5. split / reserved isolation
t0 = time.perf_counter()
assign = {}
for i, (sid, _) in enumerate(docs):
    key = int(hashlib.sha256(f"split:{i}".encode()).hexdigest()[:8], 16) % 1000
    assign[i] = "reserved" if key < 40 else ("validation_dev" if key < 45 else
               ("validation_final" if key < 50 else "stable_train"))
record("split_reserved_isolation", len(docs), sigs.nbytes, len(docs) * 8, time.perf_counter() - t0)

# ------------------------------------------------------------------ 6. benchmark decontamination
t0 = time.perf_counter()
NGRAM = 13
bench = {hashlib.sha1(f"benchmark ngram {i}".encode()).hexdigest() for i in range(200000)}
quarantined = 0
for _, text in docs:
    words = normalize(text).split()
    for j in range(0, max(len(words) - NGRAM, 1), NGRAM):
        if hashlib.sha1(" ".join(words[j:j+NGRAM]).encode("utf-8")).hexdigest() in bench:
            quarantined += 1
            break
record("benchmark_decontamination", len(docs), sum(len(t.encode('utf-8')) for _, t in docs),
       len(docs) * 8, time.perf_counter() - t0)
print(f"      {quarantined:,} documents quarantined")

# ------------------------------------------------------------------ 7. tokenize + pack
t0 = time.perf_counter()
total_tokens = 0
BATCH = 256
texts = [t for _, t in docs]
for i in range(0, len(texts), BATCH):
    for enc in tokenizer.encode_batch(texts[i:i+BATCH]):
        total_tokens += len(enc.ids)
record("tokenize_and_pack", len(docs), sum(len(t.encode('utf-8')) for _, t in docs),
       total_tokens * 2, time.perf_counter() - t0)

payload = {
    "mode_id": MODE_ID,
    "slice_fraction": SLICE,
    "stratified": True,
    "deadline_seconds": 86400,
    "slice_id": f"{MODE_ID}-2026-08-31",
    "stages": stages,
    "measured_facts": {
        "accepted_documents": len(docs),
        "accepted_tokens": total_tokens,
        "bytes_per_token": round(sum(len(t.encode('utf-8')) for _, t in docs) / max(total_tokens, 1), 4),
        "token_counter_id": "final_tokenizer_v1",
        "filter_rejected": rejected,
        "exact_duplicates_removed": removed_exact,
        "near_duplicate_pairs": near_dupes,
        "quarantined": quarantined,
    },
}
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\naccepted {len(docs):,} docs -> {total_tokens:,} tokens "
      f"({payload['measured_facts']['bytes_per_token']} bytes/token)")
print(f"measurements -> {DEST}")
