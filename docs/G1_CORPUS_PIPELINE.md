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
- A resumed state must match acquisition, source, filter, dedup, and tokenizer identities.
- Decontamination decisions commit in restart-safe batches; a failed partial batch rolls
  back while earlier complete batches remain resumable.
- Accepted text and the text-free decision ledger publish together through one sibling
  staging-directory rename. Their shared target directory must not already exist.

## Evidence

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
