# G1 production corpus pipeline

`scripts/prepare_corpus.py` is the restartable G1-04 entry point. It has six ordered stages:
acquire/filter, global deduplication, benchmark-index construction, benchmark
decontamination, boundary assignment, and atomic publication.

## Safety and determinism

- `configs/data/acquisition_v1.yaml` freezes physical-source routing, source identity,
  hash ranking, selection priority, validation quotas, MinHash indexing, restart state, and
  disk floors.
- Source text is streamed from the exact revisions in `sources_v4.yaml`; the cache path is
  mandatory and explicit.
- Filter evidence and final-tokenizer counts are stored per candidate in SQLite.
- Exact, mirror, and 128-row MinHash state is disk-backed. Thirty-two bands of four rows are
  a complete candidate index at the frozen 0.85 threshold: a passing pair can differ in at
  most 19 rows, so at least one four-row band must match.
- Connected near-duplicate components are merged and receive one canonical representative.
  Boundary selection happens afterward and assigns only that representative.
- The benchmark index stores normalized short texts and distinct 13-word shingles. A
  50-word contiguous match necessarily contains a 13-word match, so no separate 50-word
  index is needed. Hash hits are candidate detectors; exact normalized text/word checks make
  the final decision. Document n-grams are the forced outer side of short-text lookups so
  SQLite performs targeted primary-key probes instead of rescanning the benchmark index for
  every document.
- The active `decontam_v3.yaml` requires 13 normalized words in a complete matched field.
  Standalone answers such as `2` no longer trigger quarantine. The 50-word overlap and
  shingle-coverage rules are unchanged. This also leaves short copied questions and
  paraphrases outside the detector's guarantees. V2 remains frozen historical evidence.
- A resumed state must match acquisition, source, filter, dedup, and tokenizer identities.
  Decontamination additionally binds the rule digest and benchmark-input hash; unbound
  legacy decisions and mismatches fail closed, including before CLI assignment/publication.
- Decontamination decisions commit in restart-safe batches; a failed partial batch rolls
  back while earlier complete batches remain resumable. Each bounded read batch closes
  before writes so a full-scan cursor cannot pin the SQLite write log across commits.
- Accepted text and the text-free decision ledger publish together through one sibling
  staging-directory rename. Their shared target directory must not already exist.

## Evidence

Use `scripts/fork_decontamination_v3.py` only after stopping the source writer to recover
an unpublished v2 run. It takes a consistent SQLite backup (including committed WAL pages)
into a new staging directory, validates it, removes only the copied decontamination rows,
binds v3, records source-snapshot and destination hashes, and publishes the new directory.
The source state is retained. Existing destinations and any downstream selections are
refused. Legacy v2 protocol identity must be explicitly attested because old states did
not record that binding. The immutable benchmark index can be reused specifically from
v2 to v3 because normalization, source fields, and indexed shingles did not change;
the index keeps its original v2 digest. Other protocol mismatches remain errors.

See `REPRODUCING_THE_CORPUS.md` for the migration and resume commands.

The pipeline evidence JSON is `PASS` only for an end-to-end run whose selections reach all
targets, whose accepted corpus and decision ledger are published, whose every accepted
filter record has a dedup result, whose every kept cluster has a decontamination result, and
whose isolation evidence covers all four boundaries and protected slices with zero
violations. Partial stage invocations remain `NOT_RUN`.

Evidence also records measured per-stage and per-source elapsed time and counters, hashes
the checkpointed SQLite state, and hashes both published JSONL files. Those measurements
are the inputs to the required 1% and 2–5% capacity forecasts; they are not extrapolated in
advance.

The decision ledger retains source-manifest filter decisions, dedup reason/cluster/match,
benchmark quarantine evidence, and final assignment for every distinct candidate. Repeated
identical source identities are counted as acquisition-time exact duplicates rather than
silently copied.

The full scan is intentionally guarded by `--confirm-full-scan`. It remains blocked until
the measured 1% and 2–5% forecasts select `full_v1` or the dated `degraded_v1` fallback.
