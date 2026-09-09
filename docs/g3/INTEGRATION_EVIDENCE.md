# Reduced G3 section 6 integration evidence

Updated 2026-09-08. **Local implementation and correctness verification: PASS.**
**Reduced recipe readiness: `REDUCED_BASELINE_RECIPE_BLOCKED_FOR_G4`.** The required
container and lint checks were attempted but unavailable. Canonical G3/G4/G5/G6
remain `NOT_RUN`; no main baseline, full benchmark, final-holdout scoring, release,
package installation, commit or push occurred.

## Final evidence and source custody

Use only the final-v2 successor for the recovery decision:

| Artifact | SHA-256 |
|---|---|
| [Reduced G3 aggregate](../../runs/verification/g3-section6-integration-final-v2/reduced_g3_report.json) | `a587cca114ee8e6444770117ab2a96c2f3ee1b21aaf0c524c03ff8cf840209ef` |
| [Fresh postverification](../../runs/verification/g3-section6-integration-final-v2/integration_report.postverified.json) | `43f69a23d11d52991ec46a35a8918d846e9c66da4636321740e8a8a443919188` |
| Raw final-v2 report (not trusted for the decision) | `48a9957e5e6499719b8f8f22aec3fbd5170f73433003a461256965016109ca67` |
| [Final source/test manifest: 103 files](../../runs/verification/g3-section6-root/final-source-manifest.json) | `8ebf3d6ec5f863c94a0807bd1967d86841d9d25f4c5c7b28928a8153bba5dbdd` |
| Runtime production-source manifest: 65 files | `f700b854ff06bde71426b3e8c08e4d5fcbc6b56274cc822ac4463d1d8e54869f` |
| Final2 wheel, 256,296 bytes; all 32 modules byte-matched | `1e8d93d924a2c824f565326b56d0db53e52b6f76a6c19e20d66f995a255f39c5` |

The aggregate binds the first five sections' actual input/evidence files, frozen config
and source hashes, checkpoint payloads/manifests, timing history, test/build output,
evaluation smoke, environment and reviewer receipts. All 157 recursive recovery checks
are true. The postverifier reloads evidence and compares it with the current inputs/source;
it does not trust a raw PASS label. Missing or failed evidence cannot become PASS.

The [artifact index](../../runs/verification/g3-section6-integration-final-v2/artifact_manifest.json)
also binds the finalized handoff documents, review receipts and tracked working-tree patch.

Outer runner identity is `baseline-1b5417107e64091d`; the distinct training identity is
`run-5bbb7042d376832e`. Each checkpoint binds its own runner manifest's actual byte hash.
The 103-file manifest includes untracked G3 code and tests, which a Git diff alone omits.
Branch remains `codex/g3-baseline-recipe`, HEAD
`f643e9f09c0700686222712eb58d90e4a0fe1de2`, with uncommitted changes.
The frozen contract's historical PENDING fields and earlier evidence's source hashes are
superseded by this external report; their original bytes were not rewritten.

Reverify the retained artifacts without training:

```powershell
.\.venv\Scripts\python.exe scripts/run_g3_integration.py --postverify --output-root runs/verification/g3-section6-integration-final-v2
```

The executed rehearsal command used that fresh output root with `--max-updates 8
--interrupt-at 4`. A reproduction needs a different, nonexistent output directory.

## What passed

The final model has 49,658,368 parameters and used BF16, fresh seed 1337, the actual
composite exposure/development inputs, batch 8 x 32 x 1,024, and the unchanged planned
3,815-update WSD horizon. One fresh lineage stopped at 8; another stopped at 4 and resumed
to 8. Both endpoints consumed 2,097,152 loss tokens with cursor 2,048. The retained
four-update snapshot consumed 1,048,576 tokens at cursor 1,024.

Exact comparison covers model and optimizer tensors, RNG/NumPy/scaler state, best-validation
state, counters, frozen hashes, run semantics, cursor and all eight ordered input hashes.
Only explicitly documented command-location/stop fields differ. Source/manifest substitution,
command-output tampering, missing phases, incorrect counters and stale identities fail closed.
Deliberate corruption failed with `CHECKPOINT_CHECKSUM_MISMATCH`; the early endpoint failed
with the expected export-ineligibility reason. A separately identified tiny completed-decay
fixture verified the positive clean-export/reload path; it is not a completed baseline.

Full-development monitoring scored all 753 sequences / 771,072 targets in 95 batches,
including the last partial batch. The resumed 4-to-8 segment correctly inserted no extra
validation event. Validation-final was not scored. The section-5 final-v2 smoke bundle was
reopened and verified, not rescored: all five tasks have one example and the binding remains
`cb75ff8713a14244279dc62683840b559ec212e9aa03133da8316ca830351ad7`.

## Efficiency changes and measured scopes

- Detached loss values accumulate on-device, reducing logging scalar transfers from 32
  to one per optimizer update without changing optimizer math. No measured speedup is claimed.
- All best/latest/completed checkpoint writes and completed copies are timed; duplicate
  latest writes at the same update are avoided.
- `phase_timing_history.jsonl` retains each completed invocation. The interrupted timing
  and history are copied before resume, so the 0-to-4-to-8 record remains independently verifiable.
- The harness stops immediately when a required child fails and preserves a failure receipt.

| Invocation | Optimizer s | Validation s | Checkpoint s | Trainer-main wall s | Outer launch-to-exit s |
|---|---:|---:|---:|---:|---:|
| Fresh to 8 | 41.628 | 4.335 | 2.995 | 73.228 | 103.054 |
| Fresh to 4 | 20.474 | 4.229 | 3.180 | 52.173 | 106.636 |
| Resume 4 to 8 | 21.214 | 0.000 | 2.678 | 49.126 | 100.338 |

Trainer wall starts inside `train.main()` after imports. Outer command wall includes runner
preparation, child startup/imports and teardown. Their difference is not pure preparation.
The three outer training commands sum to 310.028 seconds; the raw `total_wall_seconds`
field means that sum, not complete rehearsal/campaign cost. The separately measured initial
prepare was 23.896 seconds, early eligibility command 45.918 seconds, positive fixture
command 4.118 seconds, and fresh postverification 36.934 seconds. In-process verification
and artifact-writing costs are not a separately measured end-to-end total. Baseline export
and full evaluation are NOT_RUN, and fixture wall is not isolated export timing.

The final rehearsal trained 4,194,304 engineering tokens across its two lineages and used
83.316 optimizer seconds across the three invocations. Earlier engineering attempts remain
separate history and must be included in any later complete project-cost disclosure.
Peak allocated VRAM was 5.445 GiB. Updates 2-8 ranged from 50,989 to 51,287 tokens/s,
versus the historical reduced profile's 66,989.59 weighted rate. This is a material G4
profiling question, not a sustained profile or an improvement claim. Conditional optimizer
forecasts are roughly 5.4 versus 4.15 hours for 1B, before other costs.

## Checks, repairs and independent review

Final2 full suite: **675 passed, zero failures/errors/skips**, 108.49 seconds. Compilation,
wheel build, byte comparison of all 32 packaged modules, eligibility audit, real-input opening,
and diff checks passed. Evidence is under `runs/verification/g3-section6-root/final2/`.
The current environment passed 98 checks on CPython 3.12.6, torch 2.5.1+cu124 and
RTX 4070 SUPER; see [ENVIRONMENT.md](../ENVIRONMENT.md).

The two historical permission failures were repaired by scoping implementation and source
checks to their intended repository surface. Ignored root `runs/` output cannot satisfy an
implementation marker or alter the source fingerprint; unreadable in-scope source still
fails. No protected history was deleted or made readable. Regression tests cover this.
A late compatibility edit had also made the composite manifest assignment unreachable;
a direct composite-opening test and actual-input preflight now cover that branch.

Luna built the initial rehearsal; Sol hardened the verifier/timing and repaired the composite
regression; Terra repaired audit scope and independently reviewed Sol's changes. Sol reviewed
Terra's changes. Root ran the final full suite/build/rehearsal/postverification and independently
checked files, metrics, protected inputs and wheel contents. Sol and Terra approved the same
final source/evidence with tool blockers:

- [Sol receipt](../../runs/verification/g3-section6-root/sol-final-review.json):
  `fa71f151be3d615871034dfe93ba8a99602a91f747dcb4ea916e6c8c9d649648`.
- [Terra receipt](../../runs/verification/g3-section6-root/terra-final-review.json):
  `75c67caefebaa2c6fcc56402e6e7232668dce61a4ac9e0a31c59f93a06154b4e`.

## Blockers and next task

Docker build failed to connect to `dockerDesktopLinuxEngine`; no container tests ran.
Ruff was absent; no lint PASS is claimed. The G4 verification-environment owner must
provide a working approved environment, run the documented Docker build/test and lint,
and resolve findings against the same source manifest. No host packages were installed.
These blockers keep the aggregate below READY despite local correctness passing.

Use [G4_HANDOFF.md](G4_HANDOFF.md) for exact dry-run/launch/resume/export/evaluation
commands, resource accounting, the one changed-input sustained profile, applicable
recovery/takeover check and teammate approval. G4, main training, full evaluation and
public release remain separate tasks.

Earlier `g3-section6-integration` contains a successful exactness rehearsal with incomplete
original timing/source custody; `g3-section6-integration-final` failed before GPU training
and produced no checkpoints. Initial import/identity failure directories also remain.
Only `g3-section6-integration-final-v2` is current. Data, runs, checkpoints and report files
are ignored local artifacts and require separate transfer or reproduction.
