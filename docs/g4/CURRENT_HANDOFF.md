# Current G4 handoff

Publication authorized 2026-09-12T13:19+08:00 to the existing
`origin/codex/g4-selected-recipe`; five section-3 documentation files only.
Earlier uncommitted/no-push statements below describe the verification checkpoint.

Updated 2026-09-12T13:09+08:00. **Section 3 PASS on the RTX 4070 SUPER source
lane; section 4 is next. G4 itself remains incomplete.**

Branch `codex/g4-selected-recipe`, HEAD
`6293354814a619c7b318b9d15b694c83d871083c` (published). Section-3 changes are
uncommitted documentation: PROFILE_EVIDENCE, README, STATUS, this handoff and
continuity. No project source/config changes; unrelated build/egg-info/RULES/ZIP
residue preserved. No commit/push or later-section execution occurred.

The single profile exited 0 at 2026-09-12T13:05:04+08:00; no training remains.
Evidence: `runs/verification/g4/section-03/20260912-primary-01`, with sibling
preflight/plan directories, `postverification.json` and `evidence-manifest.json`.
43-file manifest SHA:
`883d04403fda6560175294788fc66bedf278ea6fd8de162ef9ea21dd7cdcb0fc`.
17 postverification checks PASS; 133 payloads and 172 tested source files unchanged.
Prior Docker826/Windows826 tests, lint/compile/build remain applicable.

Selected CONTROL v2, fresh seed1337, BF16, 49,658,368 parameters, LR0.0006,
3815-update horizon unchanged. Runner `baseline-f23398e32dc3100e`, training
`run-146ddcb9c11da742`. Completed590 updates/154,664,960 tokens/cursor151,040.
Valid window2243.512s, weighted64,498.66tokens/s, deviceheadroom31.827%.
RAM availability briefly98.574MiB; pagefilepeak6.489GiB. Desktop workloads
remained; historical51k cause unresolved. See [evidence](PROFILE_EVIDENCE.md)
for clock definitions, overhead, headroom/thermal limits and exact hashes.

Section-4 checkpoint: `20260912-primary-01/train/latest.pt` under the evidence
root above; SHA
`efcfdd3045d097474a69299e69711e3bd4e2dd776010dada84bad2a41f7562dc`.
Carry its sidecar, identities, provenance, configuration, metrics and phase history.
This engineering checkpoint is not a completed-baseline export or G5 initializer.
Ignored evidence requires separate transfer; Git alone is insufficient.

Next safe command: `Get-Content docs/g4/04-local-recovery.md`. Bind recovery
scenarios explicitly to selected v2 (old Python harness defaults retain v1).
Current-source resume/crash equality, target profile/takeover, measured budget and
two-person approval remain. No automatic next-section launch.
