# Preparation and implementation checklist

Read [candidate specification](CANDIDATE_SPEC.md) first. This is implementation work
to be performed after the proposal is adopted; none of the new artifacts below exists
merely because its proposed path is listed.

## P0: scope, isolation and provenance

Create a successor scope that explicitly permits one fresh 3B candidate and expanded
training data. Preserve old contracts, approvals, input directories and baseline weights.
The new scope must state development-only selection, one later benchmark evaluation,
single-machine ownership, recovery/backup rules and how it supersedes the old 1B limit.
G4 owner approval for the old bundle does not approve a different bundle's bytes.

Proposed artifact map (names are reserved design choices, not executable inputs):

| Proposed path | Purpose |
| --- | --- |
| `configs/campaign/competitiveness_v1.json` | Accepted comparison scope, baseline identity and adoption rules |
| `configs/training/control_3b_v1.yaml` and digest sidecar | Complete candidate recipe and expected counters |
| New versioned acquisition/filter configs if needed | Changed data semantics with explicit inheritance |
| `data/pipeline/control_3b_v1/` | New acquisition state, selection decisions and corpus reports |
| `data/shards/control_3b_v1/` | New training shards/manifests; preserved dev identities |
| `data/schedules/control_3b_v1/` | Materialized components and verified order/accounting |
| `runs/competitiveness/control_3b_v1/` | Immutable preparation bundle and separate training run directory |
| `runs/verification/control_3b_v1/` | Code, data, environment, profile and recovery evidence |

Before writes, resolve each target within the workspace, check it is new or belongs to
the exact journaled attempt, and refuse collisions. Do not overwrite historical
`reduced_5pct_v1`, `reduced_baseline_v2`, `tokenizer_final`, G5/G6 evidence or backups.
Preserve the existing uncommitted G6 fix and unrelated build/metadata/RULES/ZIP files.

## P1: audit and data preparation

1. Inventory accepted documents, source revisions, acquisition cursors, dedup state,
   split assignments and existing benchmark index identity. Record reusable evidence
   separately from evidence invalidated by new documents or filters.
2. Select 100 training documents per source with a frozen random seed and record IDs,
   sampling frame, hashes and rubric. Report boilerplate, repeated content, extraction
   defects, language and useful-content rates. Do not use benchmark examples to guide it.
3. Specify actual filter thresholds using that training-only audit. Preserve mathematical
   notation and dialogue; report rejected counts and tokens by reason/source. Use a
   separate random 100 documents/source for the post-filter audit.
4. Process a bounded source sample, measuring bytes, wall time, accepted tokens and
   RAM/disk peaks. Report observed retention and extrapolation uncertainty. Estimate
   full acquisition/tokenization/dedup time and space before the long job.
5. Acquire from pinned revisions into isolated state, using per-source clean-token
   targets from the candidate specification. Resume only exact matching journals.
   Retry policy must not reset the token budget or fetch a different revision.
6. Deduplicate across old and new accepted material; preserve the original dev/final
   boundaries and exclude their exact/near-duplicate documents from expanded training.
   Do not scale the old dev set along with training. Do not open final holdout for
   scoring. Existing split manifests/indexes can supply exclusion identities.
7. Apply the frozen benchmark decontamination identity to all newly accepted documents.
   Changing benchmark definitions requires reviewed successor decontamination before
   data is used; never weaken exclusions because retention is low.
8. Tokenize with the original tokenizer. Build new shards with provenance, boundaries,
   EOS, source labels and hashes. Verify full stream checks, target-position uniqueness,
   quotas, source-specific reuse and data isolation. Emit a coverage/accounting receipt.

Missing a source minimum, ambiguous provenance or excessive reuse fails the data gate.
Return a specific shortage report and revised preparation estimate; do not borrow from
reserved pools or silently lower the target. A dataset labelled `FINAL` or a verifier
exit code alone is insufficient if its thresholds still refer to the historical scope.

## P2: required software changes

| Existing code | Observed limitation | Required successor behavior |
| --- | --- | --- |
| `scripts/expand_reduced_corpus.py` | Fixed 5% scope, old paths, 500M minimum and 550M target | Separate versioned expansion path; new per-source targets and preservation of old held-out assignments |
| `scripts/prepare_corpus.py` | Fraction scales existing source/validation targets; stages mutate state and may acquire data | Bind candidate acquisition/filter identity; avoid repartitioning/resizing the comparison dev set |
| `scripts/build_reduced_exposure.py` | Hardcoded old component quotas and 550,094,903-token denominator | Candidate-derived quotas, actual distinct-position accounting, correct two-component construction |
| `src/tinybench_lm/exposure.py` | Accepts only historical baseline recipe digests | Version-aware candidate identity validation, without accepting arbitrary hashes |
| `scripts/run_reduced_baseline.py` | Only v1/v2 config paths accepted; defaults select old baseline | Explicit new contract support or dedicated entry point; no changed default launch |
| `src/tinybench_lm/baseline_contract.py` and trainer identity path | Historical contract/counter assumptions | Bind candidate horizon, data, LR, selection and resume semantics end to end |
| Existing selection/reporting tools | Old screen criteria or global-only baseline logs | Complete baseline/candidate dev slice report and exact 2%/1% adoption rule |

Prefer small version-aware changes or a dedicated candidate entry point over weakening
old guards. Inspect all callers before deciding; this table is a starting map, not proof
that these are the only affected files. Reuse generic training/model code where valid.

Existing `build_schedule.py --quota` is a building block, not a verified 3B orchestration
command. Existing `prepare_corpus.py` has no plan-only switch: do not invoke it for a
“dry run.” `--help` is safe. Never change frozen YAML in place or patch old expected hashes.

## Required verification before freeze

- Unit/integration tests: exact update/token/quota/phase arithmetic; final LR zero;
  off-by-one target/cursor checks; all four slices and conservation of scored tokens.
- Negative tests: old/new config substitution, tokenizer/source/schedule drift, absent
  approval, duplicate/overlapping target reuse, insufficient corpus, early export,
  benchmark-influenced selection and changed-horizon resume must fail closed.
- Fresh CPU container build, full suite, Ruff and package/import checks. Retain logs
  tied to the reviewed source, and explain any failed check instead of omitting it.
- Actual CUDA/new-input checks: boundary reads and shifted targets, one finite update,
  precision and memory headroom, baseline dev reproduction, and candidate per-slice coverage.
- Separate bounded scratch runs: uninterrupted 8 updates versus 4 plus resume to 8;
  compare model/optimizer/RNG/cursor/counters and dev state under the same recipe.
  Test checkpoint/log-tail rollback, duplicate retry and corruption rejection.
- One sustained new-input profile with declared warmup exclusion and complete timing.
  Recalculate training, data, backup and evaluation estimates; do not use the old rate
  as measured proof for changed shards or code.
- Reverify old baseline export/tokenizer after preparation and verify every new bundle
  entry. Record source commit plus uncommitted patch/hash manifest if applicable.

Before P3, emit the concrete files named in the [runbook](RUNBOOK.md), with exact
paths, hashes, controller arguments, expected counters and command verification.
Implementation complete is not the same as measured readiness.
