# G3.5 C0/C1 return review

Reviewed 2026-09-12T10:22:06+08:00. Outcome: valid completed confirmation, CONTROL selected.

## Findings

No blocking artifact-integrity or recipe mismatch was found. Fresh analysis of the
original screening bundle plus the returned C0/C1 evidence reproduces the teammate's
entire selection.final.json exactly as a parsed JSON object.

C1 does not meet the predetermined adoption criteria:

| Measure | Observed | Requirement | Result |
| --- | --- | --- | --- |
| Global NLL reduction | 0.2891037253% | At least 0.300% | Fail |
| Broad-general NLL regression | 1.3607838739% | At most 1.000% | Fail |
| Narrative NLL regression | 2.1181334599% | At most 1.000% | Fail |
| Educational NLL regression | -1.0149991478% | At most 1.000% | Pass |
| Math NLL regression | 0.1448431352% | At most 1.000% | Pass |

C0 global NLL is 5.0516607270614395; C1 is 5.037056187709489.
The correct selected settings are LR 0.0006 and base mixture (70/20/7/3).
This is an engineering selection outcome, not a significance claim or a failed run.

## Verification and execution history

- All five completed screen/confirmation endpoints reverified against current source,
  trusted production inputs, deterministic identities/schedules, durable checkpoints,
  metric hashes, counters and full-dev slice coverage.
- C0/C1 each completed 382 updates / 100,139,008 loss tokens, seed 1002,
  batch 8 x accumulation 32, BF16 and final LR zero. Only mixture differs.
- Both recorded validation at updates 1, 100, 200, 300 and 382; endpoint coverage
  is 753 references / 771,072 targets across all four protected slices.
- Both full jobs exited zero on the same RTX 3070 Ti 8 GiB. Returned environment
  report contains 100 passing checks. This was inspected, not rerun on her machine.
- Initial C0 smoke correctly failed the memory check at 9.8267% headroom.
  The preserved retry has 12.3413%; C1 smoke has 12.3047%. The recipe was unchanged.
- All five invocation receipts reconcile with the returned runtime ledger, including
  the failed smoke. Recorded total is 4,451.469 seconds (~74m11s), including runner
  verification. C0/C1 full invocation times are 2,161.781 / 2,162.250 seconds.
- The return is a partial bundle, as expected for C0/C1: it needs the original
  bundle metadata/schedules and S0/SLR/SMIX evidence for final recomputation.
  A separate verification bundle uses hard links to the original artifacts;
  neither the return nor canonical experiment bundle was overwritten.
- Docker was unavailable. Artifact verification used the existing Windows Python
  environment on CPU, with protected-input read access. No package installation,
  training, source changes, benchmark scoring, commit or push was performed.

## Evidence and next state

Fresh evidence: runs/verification/g3.5-c0c1-return-20260912/
(selection.reverified.json, C0.smoke.reverified.json,
C1.smoke.reverified.json, receipt-audit.json).

Experiment selection is complete. Preserve the failed-attempt history and keep the
teammate ledger separate from the local lane ledger. Next is selected-settings
integration and remaining G4 sections 3–7 before fresh G5 training. This review
alone does not pass G4 or authorize a training launch; experiment weights must not
initialize the main run.
