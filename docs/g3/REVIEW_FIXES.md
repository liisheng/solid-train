# G3 pre-G4 review repairs

Repair work on 2026-09-09, branch `codex/g3-baseline-recipe`, after committed checkpoint
`9078430`. Three implementation agents handled R1–R3; a separate agent reviewed the
combined source while root checked tests, build contents and frozen input identities.

## Repaired behavior

- **R1, crash recovery:** startup validates the metrics ledger against the checkpoint and
  retains its canonical prefix. Superseded records are archived before atomic replacement,
  with invocation linkage and recorded token/optimizer costs. Retry is idempotent; a
  conflicting interrupted rollback fails closed. Malformed or incomplete logs remain
  untouched and block training. New timing records link to invocation IDs. Startup scans
  add no per-update file scan. Work lost before logging still needs external wall-time
  accounting.
- **R2, full evaluation:** full and smoke are exclusive. Full forbids limits and requires
  all five tasks exactly once (order may vary). Verification checks execution mode,
  independently pinned dataset revision/split/count, reported counts and complete unique
  document IDs. Measured pinned split sizes are HellaSwag 10,042, ARC 2,376, PIQA 1,838,
  Winogrande 1,267 and filtered WikiText 2,891. Measurement loaded datasets but ran no
  model scoring. Partial/legacy bundles do not qualify as full evidence; successor
  protocols need reviewed full-coverage registration.
- **R3, fixed identity:** runner checks normalized baseline bytes against the trusted
  digest, rejects modified parsed settings and checks the evaluation digest declared by
  the baseline. Evaluation v2 also has an independently trusted registry digest.
  Rewriting YAML and its sidecar cannot substitute the fixed version. Verified digests
  enter the runner identity; intentional recipe changes need a successor version.

Both frozen config hashes are unchanged: baseline `1d4901c4c8d83c4c4ffd17e953de2b23f5a762f2af8eb8bb57fbc6fd8e26c541`,
evaluation v2 `60eb9750bd70e74e2d5db868af068bf5cf2c578704cc1da2be12b9f2b536976b`.
The new identity fields deliberately change runner IDs. Prior runner manifests are not
silently resumable under the new identity. Existing section-6 reports remain historical;
none was overwritten or relabeled as verification of these repairs.

## Verification

- Root full suite: **715 passed**, zero failures, 124.08 seconds (`pytest.txt`).
- Independent reviewer: **90 focused tests passed**, 44.20 seconds
  (`independent-tests.log`). Both concerns found during review were resolved: coverage
  now has independent trust, and interrupted archive retries reject conflicting boundaries.
- Compilation, whitespace checks and isolated wheel build passed. All 34 wheel modules
  byte-match source; wheel SHA-256
  `4490ad1e5929d0d55d573fec575bb059d98ea7fc6ff01098f8e27b6773d0dfef`.
- Root source manifest covers 105 entrypoint/script/package/test files; no drift after
  the full suite. SHA-256
  `af6651cea79b899c78bd183dd8e3c08547bd887b9213c85fc9441ac11cc56bfc`.
- User RULES and pre-existing modified package metadata remain byte-identical.

Successor local evidence is in `runs/verification/g3-review-fixes-20260909/` (ignored local
files). Independent source disposition is recorded in `independent-review.md`.

Targeted regressions include checkpoint 100/log 120 with retained replay costs, interrupted
archive recovery, negative CLI/full-bundle cases and paired config/sidecar substitutions.
A small CPU-adapted `train.main` subprocess test exercises actual optimizer, checkpoint
and resume flow; its GPU guard and CUDA boundaries are replaced only in test memory.
It is not production CUDA verification.

## Remaining readiness work

Docker build still cannot connect to `dockerDesktopLinuxEngine`; Ruff is absent. These
checks are blocked, not passed or waived. G4 still needs its documented tool checks,
sustained changed-input profile, applicable recovery/takeover and teammate review.
Source changes require successor production evidence. No baseline, sustained profile,
full benchmark or final-holdout scoring was launched. No package was installed.
