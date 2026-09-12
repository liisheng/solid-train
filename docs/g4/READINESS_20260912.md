# G4 readiness review for sole-owner operation

Recorded 2026-09-12T14:45+08:00. G5 has not started. This review supplements the
historical freeze; its successor bundle requires the owner's explicit approval.

## Verified preflight

- All 494 historical freeze entries match raw SHA256; bundle digest remains
  `e5d5e32c08547692d94a5e30463cd76d633558ebb6bd363a8a90984994a17462`.
- Environment: 98 PASS, Python 3.12.6, torch 2.5.1+cu124, CUDA available.
- RTX 4070 SUPER, driver 591.86; sampled free GPU memory 10,553 MiB.
- Free physical RAM 6,355,896 KiB (6.06 GiB); D free 1,215,940,653,056 bytes
  before backup, comfortably above the 30 GiB floor. Resources require launch refresh.
- Existing Python PID37664 is a service, not a training command. Fresh G5 directory
  absent. Read-only launch plan exits0, runner baseline-f23398e32dc3100e; no resume
  or engineering stop. This refreshes production input and evaluation-loader binding.
- Sandbox-denied inspections are retained separately from successful access-enabled
  retries. No evidence of a frozen-file mismatch was found.

Evidence: `runs/verification/g4/readiness-20260912/` (environment.json,
resources-access.json, freeze-check-access.json, launch-plan.json and stderr).
The 172 measured production source files are unchanged; prior 826-test suites and
build/lint evidence remain applicable. No source changes require a new build.

## Owner statements and proposed approval amendment

The user named `D:\SWE\LLM BACKUP` and stated: "I am the sole custodian of this whole
project going forward. I accept." The latter answers the question about custody,
reviewer identity and retaining the 24-hour evaluation allowance as provisional.

Record the current user as sole custodian. For final review, propose one explicit
owner approval of the successor digest in place of the two-person G4 requirement.
Sole custody is recorded now; this approval-policy change becomes effective only
when the owner explicitly approves this package. Historical contracts stay intact.
Downstream G6 release approval policy is not changed by this G4 proposal.

## Backup and restore

Verified generation: `D:\SWE\LLM BACKUP\g4-freeze-20260912-1443`.
496 files / 7,311,558,017 bytes include the historical freeze payload and its two
manifest files. Source, backup and isolated restored copy all match SHA256.
Copy plus verification: 20.101 seconds; restore plus verification: 13.252 seconds.
Evidence: `runs/verification/g4/readiness-20260912/backup-check.json`.

This is a same-disk backup, with no disk-failure protection. The owner-provided
destination is used as requested. Restored bytes are verified; no restored-copy
training continuation is claimed. The earlier G4 exact local-resume evidence remains
the runtime recovery proof. No G5 checkpoint exists yet.

Proposed G5 cadence: versioned backup at every planned pause and completion, retaining
the preceding verified generation; unexpected interruption uses the last verified
100-update checkpoint. At a quiescent save boundary, copy checkpoint and sidecar plus
all lineage configuration, identities, provenance, metrics and invocation archives.
Hash source and copy; never copy latest.pt during an atomic save. Restore into an
isolated directory, verify hashes, checkpoint identity and counters, then inspect
the same-lineage resume plan before execution. No substitution of G4 weights.
Disk loss requires recovery from another verified source or a new fresh attempt;
no off-machine recovery guarantee is made. Rebudget after an unserviceable backup.

## Evaluation disposition

User accepts the 24-hour full-evaluation allowance as explicitly provisional until
an eligible G5 export exists. This closes the disposition question, not the missing
measurement: measured p90 and measured campaign fit remain NOT_RUN/BLOCKED.
Retain the 4.753-hour conservative training estimate, 24-hour downtime allowance and
54.057-hour provisional total. Keep one candidate and sequential machine use.
Optional longer training/comparisons remain excluded. Rebase calendar if the proposed
September13 08:00 UTC+8 start is missed. No deadline-extension claim is made.
After export, time and verify full evaluation; any three-run p90 calibration incurs
separate costs and needs a revised budget. Official evaluation-setting gaps remain
downstream blockers; benchmark results must not select training settings.

## Final approval

The successor bundle binds this review, historical freeze and fresh evidence.
Owner approval must explicitly accept the proposed single-owner G4 approval policy,
same-disk backup limitations/cadence, and provisional evaluation budget for that exact
digest. Until then amended G4 remains BLOCKED. This task authorizes readiness and
preflight only; baseline training requires a separate launch instruction.
