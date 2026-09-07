# Reduced 5% corpus expansion plan

This is a pending, isolated preparation for the accepted reduced scope in
`configs/campaign/submission_scope_v1.yaml`. It does not start acquisition, copy state, or
claim a gate result. The intended output is about 550,000,000 distinct stable-train tokens;
the reduced-scope minimum is 500,000,000. This is not an 11B-token G1 pass.

## Inputs and destinations

The immutable clone source is the completed top-up state
`data/pipeline/slice_1pct_v3_topup/state.sqlite`. Its original predecessor and all current
top-up publication, shards, schedules, recovery evidence, and profile work remain untouched.

The launcher creates only these new paths after an explicit `--execute`:

| Purpose | New path |
| --- | --- |
| Restartable expansion state | `data/pipeline/reduced_5pct_v1/state.sqlite` |
| Atomic accepted/decision bundle | `data/pipeline/reduced_5pct_v1_output/` |
| Stage logs and pipeline evidence | `runs/reduced_campaign/reduced_5pct_v1/` |
| Single-process lock | `runs/reduced_campaign/reduced_5pct_v1.lock` |

It uses the same pinned sources, benchmark input/index, final tokenizer, filters, deduplication,
and v3 decontamination protocol as the completed top-up. No source configuration is changed.

## Acquisition budgets

The launcher passes `--target-fraction 0.05`. `prepare_corpus.py` uses the persisted cursor and
consumed-token count, so each number below is a **cumulative** source pool budget; a restart does
not acquire that budget a second time.

| Sources | Pool factor | Cumulative pool tokens | Current top-up consumed tokens | Approximate remaining pool tokens |
| --- | ---: | ---: | ---: | ---: |
| fineweb_edu | 1.3 | 514,543,250 | 102,908,976 | 411,634,274 |
| dclm | 1.3 | 143,390,000 | 28,678,657 | 114,711,343 |
| reserved_wikipedia | 1.3 | 5,071,300 | 1,014,429 | 4,056,871 |
| openwebmath | 3.0 | 118,740,750 | 31,664,473 | 87,076,277 |
| narrative | 4.0 | 66,180,000 | 13,451,436 | 52,728,564 |
| reserved_textbook | 4.0 | 19,505,000 | 3,901,198 | 15,603,802 |

The 1.3 pool factor is retained for FineWeb-Edu, DCLM and Wikipedia because the completed slice
met their selection targets. Math uses 3.0, while narrative and textbook retain 4.0. These are
planning headrooms, not yield claims. The only observed nonzero work times are source-biased:
the prior top-up ingested 8,908 new documents in 334.819 seconds, deduplicated them in 358.624
seconds, and decontaminated them in 1,136.874 seconds. They must not be extrapolated as a
full-expansion duration forecast.

The current D: volume has about 1,291 GiB free. Reserve at least 40 GiB before launch: roughly
19.2 GiB for a fivefold SQLite state, 3.83 GiB while the source clone coexists, at most about
4.4 GiB for a fivefold accepted/decision bundle, plus index/WAL/staging margin. Cache growth is
not yet measured and must be recorded during the run; the frozen write stop at 10% free space
still applies.

## Reduced-scope targets

Whole-document selection may exceed these values; it cannot fall below them.

| Boundary / source | Target tokens |
| --- | ---: |
| stable: fineweb_edu / dclm / openwebmath / narrative | 385,000,000 / 110,000,000 / 38,500,000 / 16,500,000 |
| stable total | 550,000,000 |
| reserved science / textbook / wikipedia / edu-decile / math prose | 6,826,750 / 4,876,250 / 3,901,000 / 2,925,750 / 975,250 |
| reserved total | 19,505,000 |
| each validation split: fineweb_edu / dclm / openwebmath / narrative | 525,000 / 150,000 / 52,500 / 22,500 |
| each validation split total | 750,000 |

The stable targets preserve the accepted 70/20/7/3 mixture. The launcher verifies each selection
quota and both the 550M target and 500M minimum from the final pipeline evidence.

## Safe execution and recovery

Before changing anything, `--execute` checks for a running `prepare_corpus.py` or another
expansion launcher, acquires an exclusive lock, refuses existing fresh destinations, and clones
the source SQLite state with the SQLite backup API plus `quick_check`. A stale destination, a
staging database, an ambiguous lock, or an existing output bundle causes a refusal rather than
an overwrite.

The copied state journals these phases: `prepared`, `ingested`, `deduplicated`, `restored`, and
`published`.

1. In one SQLite transaction, it saves all old per-document decontamination decisions in an
   expansion-private table, clears only decontamination/assignments/selection keys, and records
   `prepared`.
2. It ingests each source group under its cumulative budget and then runs global deduplication.
   If interrupted before the phase advances, the persisted cursor makes the re-run append-safe.
3. In one transaction, it restores old individual decisions, drops the private saved table, and
   records `restored`. New representatives remain unclassified for the final complete scan.
4. The final all-stage pass uses the already-satisfied factor-four narrative/textbook group, so
   it does no extra ingestion, then performs benchmark-index verification, decontamination,
   assignment, and atomic publication.

The final all-stage pass cannot publish until every filtered document has dedup and
decontamination coverage, every selection is complete, and isolation passes. If a crash leaves
an output directory without its final evidence file, the launcher refuses to republish it; that
state must be inspected rather than guessed. A completed final evidence file is checked before
the journal advances to `published`.

Review the read-only plan first:

```powershell
.\.venv\Scripts\python.exe scripts\expand_reduced_corpus.py
```

After the profile has been reviewed and launch is authorized:

```powershell
.\.venv\Scripts\python.exe scripts\expand_reduced_corpus.py --execute
```

Only an interrupted, already-created `reduced_5pct_v1` state may use:

```powershell
.\.venv\Scripts\python.exe scripts\expand_reduced_corpus.py --execute --resume
```

## Bounded aggregate reconciliation

After the expansion pipeline passes, build a new source-tagged shard set and new schedules; do
not reuse the slice manifests, schedules, or profile as an expansion pass.

```powershell
New-Item -ItemType Directory -Force data\schedules\reduced_5pct_v1 | Out-Null

.\.venv\Scripts\python.exe scripts\build_shards.py `
  --documents data\pipeline\reduced_5pct_v1_output\accepted.jsonl `
  --tokenizer-dir data\tokenizer_final `
  --output-dir data\shards\reduced_5pct_v1 `
  --streaming --isolation-verified --token-counter-id final_tokenizer_v1

.\.venv\Scripts\python.exe scripts\build_schedule.py `
  --shard-root data\shards\reduced_5pct_v1 `
  --manifest data\shards\reduced_5pct_v1\stable_train.manifest.json `
  --output data\schedules\reduced_5pct_v1\stable_train.json --sequence-length 1024 --seed 1337

.\.venv\Scripts\python.exe scripts\build_schedule.py `
  --shard-root data\shards\reduced_5pct_v1 `
  --manifest data\shards\reduced_5pct_v1\validation_dev.manifest.json `
  --output data\schedules\reduced_5pct_v1\validation_dev.json --sequence-length 1024 --seed 1337

.\.venv\Scripts\python.exe scripts\build_schedule.py `
  --shard-root data\shards\reduced_5pct_v1 `
  --manifest data\shards\reduced_5pct_v1\stable_train.manifest.json `
  --output data\schedules\reduced_5pct_v1\recovery.json --sequence-length 1024 --seed 1337 `
  --quota fineweb_edu=700 --quota dclm=200 --quota openwebmath=70 --quota narrative=30

.\.venv\Scripts\python.exe scripts\verify_real_training.py `
  --shard-root data\shards\reduced_5pct_v1 `
  --train-schedule data\schedules\reduced_5pct_v1\recovery.json `
  --validation-schedule data\schedules\reduced_5pct_v1\validation_dev.json `
  --output-dir runs\reduced_campaign\reduced_5pct_v1\recovery

.\.venv\Scripts\python.exe scripts\profile_real_training.py `
  --shard-root data\shards\reduced_5pct_v1 `
  --train-schedule data\schedules\reduced_5pct_v1\stable_train.json `
  --validation-schedule data\schedules\reduced_5pct_v1\validation_dev.json `
  --output-dir runs\reduced_campaign\reduced_5pct_v1\profile
```

Then run the reduced aggregate check:

```powershell
.\.venv\Scripts\python.exe scripts\expand_reduced_corpus.py --verify-aggregate `
  --shard-root data\shards\reduced_5pct_v1 `
  --schedule stable_train=data\schedules\reduced_5pct_v1\stable_train.json `
  --schedule validation_dev=data\schedules\reduced_5pct_v1\validation_dev.json `
  --schedule recovery=data\schedules\reduced_5pct_v1\recovery.json `
  --profile-evidence runs\reduced_campaign\reduced_5pct_v1\profile\measurement.json `
  --recovery-evidence runs\reduced_campaign\reduced_5pct_v1\recovery\evidence.json `
  --aggregate-output runs\reduced_campaign\reduced_5pct_v1\aggregate.json
```

This check is a separately scope-bound reconciliation record, not a canonical G1 gate result.
It requires all of the following for `REDUCED_SCOPE_ONLY` PASS: the accepted-scope
sidecar hash; pipeline status; 5% target; 500M/550M stable checks; every scaled selection quota;
zero boundary/protected-slice violations; matching provenance/config digests; matching accepted,
decision, and state SHA-256 values; state bindings and decision coverage; bounded shard-integrity
checks with no failure; packed source/split tokens covering selected tokens; all four expansion
manifest hashes; three new schedule hashes plus semantic manifest/protocol/quota/cursor/rebuild
verification; actual stable 70/20/7/3 shares within a documented one-whole-document rounding
envelope; a fresh recovery/resume/corruption/export rehearsal bound to the recovery schedule;
and a new real-shard profile with at least 1,800 post-startup measured seconds, required
throughput/memory fields, the current recipe digest, expansion manifest/schedule bindings, and
per-update input hashes.

For each stable source, the helper reads the expansion state’s selected token total and largest
selected document. It first requires that a source’s observed overshoot is non-negative and no
larger than **that source’s** final document. It then compares the actual source share to 70/20/7/3
with `(observed_source_overshoot + target_share × total_observed_overshoot) /
actual_stable_total`. This is the upper bound from the source’s own final whole document plus the
actual denominator increase across sources. A merely possible large document in source A cannot
loosen source B; only an observed source-A overshoot, recorded in the report, contributes to B’s
denominator term. The report records actual share, source and total overshoot, and tolerance; it
does not use a fixed percentage allowance. The recovery check also reads both new
`uninterrupted` and `resumed` run configurations and requires their shard root, stable/validation
manifests, recovery schedule hash, and recipe digest to bind to the expansion bundle.

The existing streaming verifier is intentionally called in `FIXTURE` mode only to perform its
bounded integrity work. Its deferred fixture aggregate result is never interpreted as acceptance.
The helper supplies the reduced-scope checks above in a separately labelled report because the
frozen streaming verifier only recognizes FIXTURE and the original FULL thresholds. Changing
that frozen API would require a separately versioned reduced protocol and is out of this bounded
preparation.

The aggregate report explicitly defers, and does not claim: the original 11B-token G1 threshold;
the frozen full-scale shard profile/validation-size requirements; the original multi-experiment
G3 campaign; other-machine verification; baseline training; full fresh evaluation; public assets,
compute disclosure, video/screenshots; and human release approvals.
