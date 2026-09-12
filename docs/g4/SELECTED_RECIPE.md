# Selected CONTROL recipe integration

2026-09-12. Active milestone: `codex/g4-selected-recipe`. This integrates the final
experiment decision before G4 section 3; it does not pass G4 or launch G5.

## Decision and contract

Final confirmation retains CONTROL, LR 0.0006 and mixture 70/20/7/3. C1's global
improvement missed 0.3%, and broad-general/narrative regressions exceeded 1%.
`configs/campaign/selected_baseline_v2.json` binds the exact reviewed final report
using canonical JSON SHA-256 and all five endpoint/checkpoint/metric identities.
`configs/training/baseline_reduced_v2.yaml` is a full successor contract; v1 and its
hash sidecar remain unchanged. The successor normalized SHA-256 is
`dcd4d623a8f826f5e007ed33eea4bf2b653414c2088f4354a9b7e6e69a8a3f12`.

Model, optimizer math, tokenizer, corpus, seed 1337, BF16, batch 8 x 32 x 1024,
3815 updates, 1,000,079,360 loss tokens and WSD 38/3395/382 remain unchanged.
The new exposure binds the successor contract. Both schedule component files are
byte-identical to v1; the exposure content hash is
`c41bc538d0f53ee6bb08ec8b50165afbe7132ada38f3f03a0c0d46c44b792101`.
This changes the training identity and rejects an old exposure cursor. Experiment
or engineering weights must not initialize the fresh G5 baseline.

The runner CLI and G4 profiler default to v2. The runner's Python `prepare()` API
retains its historical v1 default for old harnesses; pass `config_path=SELECTED_CONFIG`
for v2. Explicit `--config configs/training/baseline_reduced_v1.yaml` remains available
for historical reproduction. Selected runtime manifests are checked against the
registered v2 digest and fixed settings before model allocation.

## Files and evidence custody

85 returned files were copied and SHA-256 verified into the canonical locations:

- `runs/pre_campaign/v2-advisory/jobs/C0` and `jobs/C1`.
- `runs/pre_campaign/v2-advisory/smoke/C0`, `smoke/C1`, and the failed C0 smoke folder.
- `runs/pre_campaign/rtx_3070.runtime.json`, separate from the local 4070 ledger.

The extracted return is archived at `runs/handoff/C0C1-return-20260912`.
`C0C1-return.zip` remains at the repository root because another process holds it open.
The prior review bundle and all original reports remain preserved. The transfer
receipt records pre-archive source paths; resolve its `C0C1-return/` prefix to
`runs/handoff/C0C1-return-20260912/` after this archive move.

Evidence root: `runs/verification/g4/selected-recipe-20260912/`.
The original 110 experiment source files were reconstructed from Git HEAD and
individually checked against the experiment source manifest. `experiment-source/`
contains this snapshot, with a read-only-use data junction to the local corpus.
Final analysis of the consolidated bundle ran from that original source. Do not
run historical endpoint analysis from the changed successor source: source custody
correctly rejects that substitution. `selection.frozen-source.json` records the
reanalysis; the canonical bundle now includes `selection.final.json`.

## Reproduction

After transferring the canonical final report and existing v1 exposure components:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m scripts.integrate_selected_baseline --output runs/verification/g4/selected-recipe-reproduction/integration.json
.\.venv\Scripts\python.exe -m scripts.run_reduced_baseline prepare
```

The first command verifies the pinned report and prepares the v2 exposure without
overwriting a conflicting artifact. Full endpoint reanalysis requires the original
experiment source snapshot. The second verifies production identities and prints
the fresh-run command. Neither command trains. The generated v2 exposure resides at
`runs/reduced_campaign/reduced_baseline_v2`; its future run directory is `run/`.

## Verification and applicability

Completed 2026-09-12T12:13+08:00:

- CPU Docker: 826 passed in 97.05 seconds; isolated Windows: 826 passed in 194.52 seconds.
- Ruff, compilation, Docker/wheel build and dependency checks pass.
- 172 container source/input files, 38 installed modules and 38 wheel modules match
  the tested source. `source-manifest.json` binds the frozen 247-file verification copy;
  later documentation-only updates are separate from the code/config/test match.
- Production preparation: `baseline-f23398e32dc3100e`; selected contract and exact
  component pins verified. CPU batches at 0, 537108 (cross-boundary), and 976632
  advance to 8, 537116 and 976640 respectively. No training/run directory was created.

See `verification.json`, `source-custody.json`, `integration.json`, `prepare.json`,
`input-reads.json`, `tests-clean.log`, `docker-tests.log` and `ruff.log` in the
evidence root. Earlier failures remain: temporary-directory permissions, source
mutation during the first experiment recheck, and an interrupted full Windows run
whose existing audit scanned large local data files. The isolated final suites
supersede these attempts. Docker required both stale runtime socket directories to
be preserved together before restart; no image reset or host package installation.

Unchanged component bytes preserve the existing data-order and input-reader proof.
The selected contract, runner bindings and trainer startup guard need successor
checks. Historical timing/recovery cannot stand in for current-source G4 sections
3–4. Section 3 must refresh the profiler plan, source/input custody and live telemetry.
Takeover, measured budget and two-person freeze approval remain sections 5–7.
