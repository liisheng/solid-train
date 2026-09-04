[PLANS]

- 2026-09-05T00:26:00+08:00 [USER] Complete G1 on `g0-evidence`: pin a reproducible provisional benchmark/decontamination scope, then acquire, filter, deduplicate, decontaminate, shard, schedule, and verify the real corpus without claiming unsupported passes.

[DECISIONS]

- 2026-09-05T00:26:00+08:00 [USER] Proceed with current public benchmark versions because the official GIBC V2 pages do not promise a later version specification.
- 2026-09-05T00:26:00+08:00 [USER] Install and follow the supplied root `AGENTS.md`.
- 2026-09-05T00:26:00+08:00 [CODE] Use the container only for CPU installation/tests; retain the measured Windows CUDA environment for RTX training until a GPU container is verified.

[PROGRESS]

- 2026-09-05T00:26:00+08:00 [TOOL] Merged `origin/sabrina-pc` into `g0-evidence` and pushed merge commit `74316d1`.
- 2026-09-05T00:26:00+08:00 [TOOL] Teammate contribution passed 556 tests and all nine tokenizer verification checks before merge.
- 2026-09-05T00:51:56+08:00 [CODE] Added frozen `decontam_v2.yaml`, a benchmark-input fetcher, count/hash evidence, tests, and reproduction documentation for the production G1 quarantine inputs.
- 2026-09-05T00:51:56+08:00 [TOOL] Materialized 1,362,239 usable benchmark rows at SHA-256 `976f81cfef5ef3d6ff9d3d608c482b348e1e9606acec1dfd261d2f235536fa4d`; an independent local recount and rehash matched the committed evidence.
- 2026-09-05T00:51:56+08:00 [TOOL] Full host verification passed: 560 tests in 74.50 seconds, Python byte-compilation passed, and the source-policy audit returned PASS with only accepted-token measurement correctly DEFERRED.
- 2026-09-05T05:20:14+08:00 [CODE] Replaced fixture-only production behavior with a sorted-input streaming shard path: incremental tokenization and uint16 writes, SQLite-backed global ID/cluster/shard-record indexes, token/document shard ceilings, atomic staged publication, and safe fresh-output restart refusal.
- 2026-09-05T05:20:14+08:00 [TOOL] Streaming implementation verification passed: 33 shard tests, 564 full-suite tests in 70.43 seconds, Python byte-compilation, and diff checks.

[DISCOVERIES]

- 2026-09-05T00:26:00+08:00 [CODE] Existing `scripts/build_shards.py` is fixture-oriented and explicitly does not produce billion-token shards; G1 needs a scalable production path.
- 2026-09-05T00:26:00+08:00 [TOOL] Official GIBC V2 rules name the benchmark families but do not promise later harness commits, dataset revisions, shot settings, metric keys, or WikiText scoring details.
- 2026-09-05T00:26:00+08:00 [CODE] The supplied `AGENTS.md` contained a repository-specific Docker paragraph for an unrelated FastAPI/Node project; it was superseded locally with TinyBench-LM instructions.
- 2026-09-05T00:51:56+08:00 [CODE] The alignment-auditor mirror fixture copied the multi-gigabyte `.cache` directory into C: temp space; `.cache` and `.hypothesis` are now excluded from fixture mirrors, and four exact pytest temp directories were removed, restoring C: free space from 0.76 GB to 11.08 GB.
- 2026-09-05T00:51:56+08:00 [TOOL] Docker 29.5.3 is installed, but the Docker Desktop Linux engine is not running; container verification remains unrun until Docker Desktop is opened manually.
- 2026-09-05T05:20:14+08:00 [CODE] Arbitrary input cannot be deterministically sorted or globally near-deduplicated with bounded memory; production shard input must be sorted by frozen boundary/source/document order and carry upstream isolation evidence. The builder validates explicit cluster IDs but leaves global isolation and mixture/profile reconciliation DEFERRED rather than inventing a pass.
- 2026-09-05T05:20:14+08:00 [CODE] The frozen v1 JSON manifest embeds every document boundary. Packing is bounded-memory and manifests are written from disk-backed records, but loading one complete manifest still materializes that split's boundary arrays; a future streaming aggregate verifier is needed for full production reconciliation.

[OUTCOMES]

- 2026-09-05T00:26:00+08:00 [TOOL] G0 evidence and the teammate's tokenizer/pipeline artifacts are integrated on `g0-evidence`; G1 remains in progress.
- 2026-09-05T00:51:56+08:00 [TOOL] The production benchmark inputs required for G1 decontamination are pinned, acquired, and verified; G1 remains incomplete because the scalable 11B-token corpus build, real decontamination scan, final sharding, and schedules have not run.
- 2026-09-05T05:20:14+08:00 [TOOL] The scalable shard-packing stage is implemented and verified. G1 still requires the upstream real-corpus acquisition/filter/dedup/decontamination stream, its isolation evidence, the full shard run, and streaming mixture/profile reconciliation.
