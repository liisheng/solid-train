# G6 evaluation and release

Full provisional evaluation completed on 2026-09-13. Coverage, artifact hashes and
the saved bundle passed controller verification and a subsequent root review.
**G6 release is still blocked.** See [results and timing](RESULTS.md) and
[project status](../STATUS.md).

## Completed work

- All required tasks, with no limit: HellaSwag 10,042; ARC-Easy 2,376; PIQA 1,838;
  WinoGrande 1,267; WikiText-103 2,891 nonempty test paragraphs.
- Fresh Windows environment: Python 3.12.6, PyTorch 2.5.1+cu124; hashed source
  snapshot and G5 export. CUDA/BF16, batch 16, zero-shot, seed 1234, bootstrap 1,000.
- Fixed missing document records required for full-coverage verification. Frozen
  scoring settings are unchanged. Docker build/Ruff, 76 evaluation tests and
  17 Windows coverage tests passed.
- Reconciled named G5/G6 process durations. Evaluation took 192.305 s; scoring
  inside it took 99.558 s. See the results report for accounting exclusions.

The original setup check failed on a configuration-path lookup. The corrected
environment check passed before scoring; the failure receipt remains historical.
The old 24-hour allowance is not an actual runtime or measured p90 estimate.

## Artifact identity

| Artifact | Value |
| --- | --- |
| Export | `runs/reduced_campaign/reduced_baseline_v2/run/baseline_export.pt` |
| Export SHA-256 | `89438ee3165a64bd3c15b3ba1f4aa6658841a423a554164061c2d4c0ac95c288` |
| Tokenizer | `data/tokenizer_final/tokenizer.json` |
| Tokenizer SHA-256 | `2103d520df7a23490054cff474f5c0e0f241bb56b51051284ab92889a683c635` |
| Parameters | 49,658,368; export reload and recount verified in G5 |
| Evidence directory | `runs/verification/g6/20260913-01` |
| Final receipt | `EVALUATION_VERIFIED_PENDING_RELEASE_REVIEW` |

Paths above are repository-relative. The local ignored evidence directory contains
setup/controller logs, `fresh-environment.json`, `receipt.json`, `results.json`,
`bundle/`, source hashes and both postverification reports. The completed attempt's
observer is retained at `runs/verification/g6/observe-g6.ps1`; it is not a launch
command. Do not rerun evaluation because an older setup receipt says FAILED.

## Remaining release work

1. Resolve organizer settings and harness commit provenance. Current scores must
   retain the label `PROVISIONAL_NOT_OFFICIAL`.
2. Complete the final package and total campaign efficiency accounting, including
   earlier experiments, failed attempts and verification work.
3. Obtain publication authorization, publish approved assets and verify accessibility.
4. Obtain required human approvals of the exact final release hash. G4's single-owner
   amendment did not change the G6 release policy.

Benchmark scores cannot guide training choices under the frozen contract. The
[competitiveness proposal](../experiments/COMPETITIVENESS_PLAN.md) is a separate draft.
Model binaries remain local; evaluation does not establish publication or a G6 PASS.
