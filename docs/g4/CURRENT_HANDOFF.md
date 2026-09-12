# Current G4 handoff

Updated 2026-09-12T13:44+08:00. **Section 4 PASS: local recovery. Section 5 next;
G4 itself incomplete.** Branch `codex/g4-selected-recipe`, HEAD
`1cd448ec6754ff1fe06598b324c0a341c8c42276`. Production source/config unchanged;
section-4 evidence and coordination documentation remain uncommitted. Preserve
unrelated build/egg-info/RULES/ZIP residue. No publication or later section started.

The v2 real-model BF16 eight-update versus four-plus-resume rehearsal and separate
checkpoint4/log8 rollback replay pass exact durable comparison. One superseded
archive preserves four replayed updates, 1,048,576 tokens and15.969 optimizer seconds.
Corruption/early-export rejection and archive retry fault-injection checks pass.
All90 evidence-file hashes and172 source hashes match;20 successor review checks pass.
Prior826 Docker/Windows tests, lint/build and section3 timing remain applicable.
No training remains. See [recovery evidence](RECOVERY_EVIDENCE.md) for scope and limits.

Evidence `runs/verification/g4/section-04/20260912-primary-02`:
`review.json` PASS supersedes original controller result AWAITING_REVIEW.
Manifest SHA `1fcd493d237dac5d880c011c700d36a6b7c210a9744d33055f99bb13430301cb`;
review SHA `da9e09c3f0452602b809f4d216d5ec90194536966c1bf4caff116d2229a70e78`.
Attempt01 failed before training on corpus sandbox access and remains preserved.
Four training commands consumed389.783 outer seconds; successful controller543.071s
through result writing. Rollback paths retain execution-time clean-path identities;
clean directories were restored, rollback evidence is now under `rollback/`.

Takeover candidate remains section3's verified590-update checkpoint:
`runs/verification/g4/section-03/20260912-primary-01/train/latest.pt`, cursor151040,
SHA `efcfdd3045d097474a69299e69711e3bd4e2dd776010dada84bad2a41f7562dc`.
Carry sidecar, metrics, phase history, provenance, configurations and identities.
Its bytes are unchanged; section4 proved fresh bounded recovery, not a resume of
this particular checkpoint. These engineering weights cannot initialize G5.
Selected CONTROL v2, runner `baseline-f23398e32dc3100e`, training `run-146ddcb9c11da742`.

Next safe command: `Get-Content docs/g4/05-takeover.md`. Target-machine custody,
resume/profile, measured budget and two-person approval remain. No target access
or teammate action is assumed; Git alone does not transfer ignored evidence.
