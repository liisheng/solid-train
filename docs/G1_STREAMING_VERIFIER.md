# G1-05 streaming verifier

`scripts/verify_shards_streaming.py` verifies a production shard directory without
materialising a complete `*.manifest.json` in memory:

```text
.venv\Scripts\python.exe scripts\verify_shards_streaming.py data\shards --scale FINAL \
  --isolation-evidence data\shards\isolation.evidence.json \
  --output docs\evidence\shards\g1_streaming_verification.json
```

The verifier parses one namespace and one shard record at a time. It memory-maps only the
current uint16 shard for boundary/range checks, hashes shard bytes in chunks, and keeps
cross-split document/shard/path uniqueness in a temporary SQLite index. Manifest content
hashes are rebuilt from a disk spool, so the full v1 boundary arrays never reside in RAM.

Final-scale aggregate checks require all four frozen manifests and
`final_tokenizer_v1`. Missing manifests, a provisional counter, or absent isolation evidence
remain `NOT_RUN`; malformed or contradictory evidence is `FAIL`. The verifier never treats
`--isolation-verified` or a JSON boolean as proof. Isolation evidence must be a JSON object
with `status: PASS`, zero `boundary_violations`, `slice_violations`, and `undeclared_slices`,
and explicit coverage of all four isolated boundaries and four protected slices.

The result includes source token shares, stable/reserved totals, reserved-margin status,
validation sizes, profile selection, per-shard integrity/dtype/source/boundary checks, and
machine-readable facts. It is the G1-05 aggregate stage; it does not acquire data, perform
near-deduplication, or create shards.
