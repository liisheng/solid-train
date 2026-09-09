# Independent pre-G4 review

Reviewed 2026-09-09T09:37:18+08:00. **Disposition: changes requested before a clean G4
readiness sign-off.** This review assumes the user's "phase 4" means G4 after G3 section 6.
No implementation was changed, no GPU training/profile/full evaluation was launched, and no
packages were installed or commits/pushes made.

The existing section-6 tests and retained evidence still pass. They cover a clean bounded
stop/resume and a bounded scoring rehearsal; they do not cover the issues below. The previous
local PASS is preserved as evidence of those checks, not treated as proof of every failure path.

## Findings

### R1 — P1: reconcile metrics when recovering an older checkpoint

Locations: `train.py:721-743`, `train.py:803`, `train.py:852`, `train.py:956-969`;
`scripts/run_reduced_baseline.py:216-228`.

The trainer restores the checkpoint's completed update but appends to the existing
`metrics.jsonl`, resetting record-continuity checking for the invocation. The runner checks
the checkpoint/stop target but never reconciles the existing metrics tail with that checkpoint.

Example: latest checkpoint contains completed update 100, metrics contain updates 101–120,
and the process crashes before the next save. Resuming latest deterministically replays those
updates and appends duplicate/out-of-order rows. Model recovery can be correct while the
unsegmented training ledger no longer describes a unique lineage. The production export
verifier does not inspect that ledger. The integration postverifier would reject duplicated
rows in its small fixture, but its successful 4-to-8 rehearsal starts from a clean stop.

Sol identified this path and Terra independently confirmed it. This is a control-flow finding,
not a newly executed GPU crash experiment.

Required repair: reconcile the log prefix against the checkpoint before training. Preserve
superseded/failed-attempt work in a separately identified segment or invocation ledger, and
make canonical lineage records unambiguous. Reject unexplained gaps/duplicates and seed
continuity validation from the retained prefix. Add a crash-tail regression for checkpoint 100
with records through 120, then resume; verify both lineage accounting and retained replay cost.
Do not silently discard failed work from compute disclosure.

### R2 — P2: full evaluation must reject limited or partial runs

Locations: `evaluate.py:171-180`, `evaluate.py:229-244`;
`src/tinybench_lm/evaluation_protocol.py:1421-1536`.

`--full` only prevents automatic smoke fallback. `--full --limit 1`, conflicting full/smoke
flags, and a subset of required tasks are not rejected; the requested limit reaches the
harness. Bundle integrity verification establishes internal consistency, not full required
split coverage. An operator can therefore use the full path while scoring only a small sample.

Both independent reviewers confirmed this code path. No benchmark was rescored for this review.

Required repair: make full/smoke modes exclusive; require no limit and the complete required
task set for the primary full run. Persist the mode and verify actual task/sample coverage
against the intended full splits. Keep intentionally partial scoring separately labeled.
Add negative CLI and persisted-bundle tests for limits, missing tasks and conflicting flags.

### R3 — P2: enforce the fixed recipe/evaluation digests at runtime

Locations: `scripts/run_reduced_baseline.py:92-96`, `:119-181`;
`src/tinybench_lm/evaluation_protocol.py:195-212`.

The runner derives expected recipe semantics from the live baseline YAML without comparing
its actual normalized content to the fixed recipe digest/sidecar. The exposure verifier binds
the persisted exposure to a recipe constant, but does not compare that constant to the live
runner YAML. The runner also does not enforce the evaluation-v2 digest declared in the baseline
contract. Evaluation v2 is accepted through a matching adjacent sidecar rather than an
independently trusted fixed digest, so changing both permits a new self-consistent protocol.

Terra completed an unmocked CPU identity-construction check using the actual exposure, with
only the config's in-memory peak LR changed from 0.0006 to 0.0005. It accepted
`baseline_reduced_v1`, unchanged exposure `6b1b747e47ea907810bf68f9ff941f61319ba7dbc7a034e3e2a746775666b0e3`,
and new identity `baseline-cd0d023576e8a882`. No original config was modified and no launch or
export was attempted. This proves the identity-construction gap, not a changed trained model.

The external source/artifact manifest correctly protects the currently reviewed snapshot;
all its hashes matched. It is not an automatic guard in each executable entry point. This
finding is therefore a missing runtime immutability check, not evidence of current tampering.

Required repair: check actual recipe and evaluation bytes against the existing trusted
version-specific digests before constructing identities/launching/scoring. Bind those checks
into the run record. Publish a successor version for intentional semantic changes. Test a
recipe LR mutation and a paired evaluation YAML/sidecar mutation, including transitive inputs.

## Fresh verification

- Sol reviewed training/exposure/resume/export and evaluation paths; Terra independently
  reviewed overlapping paths, protocol/evidence binding, integration verification and audit
  scope; Luna reran CPU verification. Root reviewed human contracts and checked identities.
- All 103 source/test files match final source manifest
  `8ebf3d6ec5f863c94a0807bd1967d86841d9d25f4c5c7b28928a8153bba5dbdd`.
- All 221 indexed artifacts, totaling 2,035,482,613 recorded bytes, matched their saved hashes
  before this review updated coordination documents. Historical manifests remain unchanged.
- Fresh `.venv\Scripts\python.exe -m pytest -q`: **675 passed, zero failures**, 168.35s.
- Compilation of root entry points, scripts, source and tests passed (exit 0).
- Fresh postverification reading the original final-v2 root passed, with the same result hash
  `43f69a23d11d52991ec46a35a8918d846e9c66da4636321740e8a8a443919188`. New output:
  `runs/verification/g3-section6-postverify-audit-20260909/integration_report.postverified.json`.
  Original final-v2 report was not overwritten. An initial relocated-copy check failed because
  paths/identities changed; it was not treated as failure of the original retained evidence.
- Docker still cannot connect to the Linux engine; container checks remain unrun. Ruff is
  still absent; no lint PASS is claimed. No package build was needed after a source-unchanged
  review; the earlier byte-matched build remains historical evidence.

## Next action

Fix R1–R3 and add their regressions before requesting a clean readiness sign-off. Re-run the
affected checks and integrated verification against the repaired source; retain old evidence.
Then continue G4 tool verification, one changed-input sustained profile, applicable
recovery/takeover and teammate review. The short ~51k versus historical ~67k tokens/s gap
remains a profiling question, not a demonstrated sustained regression or a speedup claim.
Do not launch G5 solely because the existing 675-test suite passes.
