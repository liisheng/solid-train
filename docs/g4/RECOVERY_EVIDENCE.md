# G4 section 4 local recovery

2026-09-12T13:44+08:00: **Section 4 PASS — local recovery only.** User authorized
section 4 and observed execution. Branch `codex/g4-selected-recipe`,
HEAD `1cd448ec6754ff1fe06598b324c0a341c8c42276`; production source unchanged.

## Declared coverage before execution

| Scenario | Evidence / planned successor |
|---|---|
| CPU regression, source/input identity | Reuse 826-test verification only after matching all 172 section-3 source hashes; production runner verifies selected input bindings again. |
| Exact CUDA recovery | Fresh BF16 real-model v2 eight updates versus four plus resume to eight; full 3815-update horizon. Existing harness exact recursive tensor/optimizer/RNG/scaler/best-state checks and artifact postverifier. |
| Crash/log rollback | Preserve the clean rehearsal, copy its resumed directory, restore its four-update checkpoint with eight real metric rows retained, resume to eight. Compare exact durable state and ordered batches; archive must retain precisely four superseded rows and costs. This simulates durable-state/log skew, not an abrupt GPU process kill. |
| Archive retry | Existing focused CPU fault-injection test covers crash between archive and atomic replacement, retry twice without duplication. |
| Corruption / early export | Harness corrupts only a task-owned checkpoint copy and requires checksum mismatch; early eight-update checkpoint must fail export eligibility. |
| Timing custody | Preserve every command, exit code, outer wall, phase history and replay cost; canonical tokens plus archived tokens are separate from total invocation wall. |

Evidence controller: `runs/verification/g4/section-04/run_recovery.py`.
Completed attempt: `runs/verification/g4/section-04/20260912-primary-02`.
Attempt 01 failed before training on sandbox access to the dev manifest; its
environment check passed and failure/timing records remain preserved.
The adapter passes selected v2 explicitly to the existing harness Python prepare
API and runner commands; it changes no trainer, checkpoint, ledger or recipe code.
The section-3 590-update checkpoint remains untouched and available for section 5.

Exact command (repository root, do not relaunch an existing attempt):

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -u runs/verification/g4/section-04/run_recovery.py 20260912-primary-02
```

## Verified result

The controller finished at 2026-09-12T13:38:00+08:00. Independent artifact review
passes all 20 checks in `review.json`: all 90 manifest entries and 172 source files
match; clean and rollback checkpoints reverify and exactly match uninterrupted
training; canonical prefixes, batch order, losses, timing linkage and archive match.
The clean harness postverification's complete check tree is true. No training remains.

All paths use selected CONTROL v2, seed1337, actual 49M model, CUDA BF16 and
the unchanged 3815-update horizon. Runner `baseline-f23398e32dc3100e`, training
`run-146ddcb9c11da742`; contract SHA `dcd4d623a8f826f5e007ed33eea4bf2b653414c2088f4354a9b7e6e69a8a3f12`,
exposure content SHA `c41bc538d0f53ee6bb08ec8b50165afbe7132ada38f3f03a0c0d46c44b792101`.
Exact comparisons include model/optimizer tensors, RNG, scaler policy/state,
best-validation state, counters, cursor, frozen hashes and run identity. The
harness excludes directory/resume/stop arguments and separately verifies runner
manifest semantics and embedded hashes; this is not an equality claim for wall time
or invocation IDs. Eight updates end at 2,097,152 loss tokens and cursor2048.

Rollback restores checkpoint4 while retaining real rows through8. One archive
contains exactly rows4–7; active rows remain indices0–7, with identical prefix,
losses and batch hashes. No duplicate active rows. Its three timing histories are
0→4, 4→8, 4→8. To preserve embedded path identities, the driver temporarily
ran the rollback copy at the clean lineage's original path, then restored the
clean directory. Archived paths therefore refer to execution-time locations;
the rollback artifacts now reside under `rollback/`. Originals remain intact.

The archive-before-replacement retry test passes (1 passed,17 deselected), including
two reconciliations without duplicate archives and rejection of a conflicting
boundary. This is CPU fault injection plus a real CUDA log/checkpoint-skew replay,
not evidence of an abrupt GPU process kill. Corrupt-copy checksum rejection and
early-export rejection pass; isolated completed-decay positive fixture passes
(1 passed,14 deselected). BF16 does not exercise FP16 scaler recovery.

| Accounting | Seconds / tokens |
|---|---:|
| Four actual training commands, outer launch-to-exit sum | 389.783 s |
| All eight recorded child commands | 444.451 s |
| Successful controller through result generation | 543.071 s |
| Failed pre-training attempt through failure recording | 3.036 s |
| Rollback lineage optimizer total, including superseded work | 48.321 s |
| Canonical rollback optimizer time | 32.352 s |
| Archived optimizer time | 15.969 s |
| Canonical rollback tokens / superseded tokens | 2,097,152 / 1,048,576 |
| All four training invocations, including uninterrupted reference | 5,242,880 tokens |

Controller timing excludes final manifest hashing, imports before its timer and
this review. Child wall includes runner preparation; phase timing starts inside
the trainer. These are nested measures, not additive costs. The failed attempt is
retained separately; its environment command is inside its controller time.

## Custody and next section

The 90-file `evidence-manifest.json` SHA is
`1fcd493d237dac5d880c011c700d36a6b7c210a9744d33055f99bb13430301cb`.
It covers the original run, copied controller source, receipts and raw comparisons;
the later `review.json` is separate successor evidence. Original `result.json`
remains AWAITING_REVIEW to preserve the controller record; this review supersedes it.
Clean eight-update checkpoint SHA:
`8d1f369f8a954cefe57d2b9472cfaec6e5ccb5f18bf5c14fa20d27d5cb75110a`.

The unchanged section-3 checkpoint remains the higher-progress takeover candidate:
`runs/verification/g4/section-03/20260912-primary-01/train/latest.pt`, update590,
cursor151040, SHA `efcfdd3045d097474a69299e69711e3bd4e2dd776010dada84bad2a41f7562dc`.
Carry sidecar, metrics, phase history, provenance, configuration and identities.
Its bytes match the section-3 evidence; this section proves recovery on fresh bounded
lineages, not a resume of that specific checkpoint. All are engineering artifacts,
never G5 initializers or completed-baseline exports. Git does not transfer ignored evidence.

No production code/config changed: section-3 timing and source-bound 826-test
Docker/Windows, lint/build checks remain applicable. New adapter/observer syntax
checks and documentation whitespace checks pass. Cross-machine takeover, budget
and two-person approval remain; G4 itself is incomplete. Next: section5 only on
request, `Get-Content docs/g4/05-takeover.md`. No commit/push in this task.
