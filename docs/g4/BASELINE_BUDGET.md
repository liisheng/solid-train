# G4 section 6 — fixed baseline budget

Recorded 2026-09-12, Asia/Singapore (UTC+08:00). **Section 6 deliverable complete;
execution budget conditional, measured campaign fit BLOCKED.** No training or scoring
was launched. Section 7 is next; G4 remains incomplete. This report paraphrases the
contracts and separates measurements, extrapolations and discretionary allowances.

## Authority and date

Selected CONTROL v2 remains fresh seed 1337, 49,658,368 parameters, BF16,
8 × 32 × 1024 batch, LR 0.0006, 3815 updates and 1,000,079,360 loss tokens.
WSD is 38/3395/382; final cursor 976,640. This is the fixed engineering recipe
retained after confirmation, not a newly optimized LR. Neither experiment nor
engineering checkpoint weights initialize G5. Budget candidate count is **one**:
the eligible completed baseline. An early best checkpoint is not another candidate.

The governing scope is `submission_scope_v1/v2` plus `g4_scope_v2`; section 5 is
not applicable. Training, export and evaluation share this RTX 4070 SUPER sequentially.
The [machine-readable calculation](BASELINE_BUDGET.json) contains formulas, units,
raw-byte source hashes, exact results, reserves and source-applicability checks.

Official rules checked 2026-09-12T13:58+08:00 still conflict: their header lists
October 1, 2026, while their dated body specifies September 21, 2026 at 23:45
Taipei time (UTC+8). Use **2026-09-21T23:45+08:00 provisionally**, pending organizer
clarification; an extension is unconfirmed. [Official rules](https://gibc-v2.devpost.com/rules).
The frozen operations contract separately says fit before September 17, without a
time/year. This calendar conservatively interprets that as 2026-09-17T00:00+08:00;
it does not rewrite the contract. September 5 freeze is historical; September 18
submission preparation remains an internal target, not an organizer cutoff.

## Measurement and extrapolation

Section 3 measured 552 post-warmup updates over 2243.512 optimizer seconds. Weighted
rate is 64,498.658839 tokens/s; per-update p10 is 60,735.301112 tokens/s. The latter
gives the slower scenario; this is not a statistical p90 of complete-run runtime.
The 590-update profile has six full-dev events, six recovery writes (including
its bounded end), and five improving best writes, re-counted from the metrics.
Its phase totals are 20.648794 seconds validation and 17.066286 seconds checkpointing.

| Component (seconds) | Central extrapolation | Conservative extrapolation |
|---|---:|---:|
| Tokens / rate, all 1,000,079,360 tokens | 15,505.429 | 16,466.196 |
| First-38 warmup excess above rate projection | 3.294 | 0.000 |
| All 40 full-dev events | 137.659 | 275.317 |
| 39 recovery writes + allowance for 40 best writes | 122.567 | 245.134 |
| Preparation, startup and residual | 91.899 | 125.347 |
| Training subtotal, excluding reserves below | **15,860.847 (4.406 h)** | **17,111.994 (4.753 h)** |

Validation uses measured total / 6 × 40. Checkpointing uses measured total / 11 × 79;
this pooled mean does not isolate best versus recovery write latency. All 40 possible
best saves are allowed, not predicted. Conservative overhead doubles those means.
Neither model measures end-of-decay throughput; measured stable-phase rate is extrapolated.

Startup uses the separately measured PLAN_ONLY preparation 54.451 s (execution
preparation was 28.258 s), trainer residual 27.897646 s, child-minus-main 5.550470 s,
and a discretionary 4 s wrapper allowance. Conservative doubles residual and launch
terms. Residual includes setup/logging and unclassified work; its fixed extrapolation
is an assumption. It is not relabeled as measured preparation. Warmup adds only the
positive difference between measured first-38 time and its rate-based projection.
Data wait is already inside optimizer time. No elapsed-window or child wall total
is added on top of these constituent phases.

## Separate reserves and recovery

| Discretionary allowance | Seconds | Basis / limitation |
|---|---:|---|
| Completed-checkpoint custody copy | 60 | Untimed production completion operation; separate from 79 writes |
| Endpoint verification, export, reload/recount | 1800 | Provisional 30 min; full eligible export not timed here |
| Backup copy, hashes and restore verification | 1800 | Provisional 30 min; destination/bandwidth unconfirmed |
| Full evaluation, one candidate | 86400 | Provisional 24 h allocation, **not measured p90** |
| One local restart and replay | 1006.433 / 1031.617 | 100 updates / scenario rate + unmeasured 600 s restart allowance |
| Machine downtime | 86400 | Discretionary 24 h; no alternate-machine capacity |
| Combined training plus all reserves | **53.702 / 54.057 h** | Conditional calendar allocation, not GPU-hours |

Section 4 proves exact local resume/rollback. Its tiny rehearsal does not measure a
full machine restoration. The replay allowance conservatively covers 100 updates
(26,214,400 tokens); a verified latest checkpoint normally bounds lost completed
work below that cadence. Multiple failures or loss of the machine can exceed this
allocation. Pause, preserve receipts/archives, restore and verify the same lineage;
re-budget if reserves are consumed. Close unnecessary background workloads before
launch and recheck memory: profile minimum free RAM was only 98.574 MiB despite
successful completion. No concurrent GPU work is budgeted.

The required evaluation formula remains **measured full-suite p90 runtime × 1 × 1.25**.
Runtime evidence is NOT_RUN; its measured reserve and measured-fit claim are BLOCKED
(`OPS_UNMEASURED_INPUT`). To fit the provisional 24 h slot, measured p90 must be at
most 69,120 s (19.2 h), or the calendar must be revised. This threshold is algebra,
not an estimate of evaluation speed.

Separate calibration task, requiring its own authorization: identify an eligible
completed artifact, verify its export/identity, and time at least three complete
sequential CUDA/BF16 evaluations in the declared fresh environment. Include all
required tasks (HellaSwag, ARC-Easy, PIQA, WinoGrande, pinned WikiText-103), exact
coverage, no sample limit, fixed batch/loader/harness/protocol, outer receipts and
failures. Derive nearest-rank full-suite p90; with three runs that is their maximum.
Charge every calibration run separately before applying the candidate reserve;
the current 24 h candidate slot does **not** fund three extra calibration runs.
No eligible artifact or calibration runtime is established here. The existing CPU
one-example smoke is unsuitable. Organizer-dependent scoring settings remain
provisional; benchmark results must not choose training settings. Section 7 must
expose this dependency and the amended scope's explicitly provisional budget option,
not invent measured readiness to resolve the G5-artifact dependency.

## Storage

Measured latest checkpoint: 596,076,206 bytes (0.555 GiB). Reserve eight equivalents:
latest/best/completed, one atomic-write temporary, three verification/backup staging
copies, and one recovery copy: 4.441 GiB. Add 2 GiB for exports, 1 GiB logs/manifests,
and 10 GiB evaluation/cache/results: **17.441 GiB additional**, rounded to a **30 GiB
free-space floor** on D. These are planning allowances; retaining every periodic
version would need a larger budget. The trainer overwrites latest rather than
retaining 39 distinct files. Existing historical evidence is not deleted.

Profile metrics are 489,964 bytes for 590 updates (about 3.02 MiB projected for 3815);
the 1 GiB allowance also covers telemetry, sidecars and retries. Export/cache sizes
are unmeasured. At this check D had approximately 1.216 TB free and C 197.3 GB;
exact byte snapshots are in JSON. This establishes current local space, not future
availability. Reserve about 4.665 GiB off-machine for three checkpoint equivalents
plus 3 GiB metadata/export allowance; input data/environment also need a separately
verified restoration source. Destination, throughput and restore custody remain
unconfirmed for section 7. A same-disk copy is not a machine-failure backup.

## Provisional dated calendar

All windows are **2026, Asia/Singapore UTC+08:00**, on this one machine. Availability
was requested from the user and is UNCONFIRMED. These are proposed reservations,
not launch authorization. Section 7 approval/backup policy must precede G5.

| Start | End | Owner | Work and dependency |
|---|---|---|---|
| Sep 12, after this report | Sep 13 08:00 | Operator + second human | Section 7 review, availability, backup policy; no invented approval |
| Sep 13 08:00 | Sep 13 13:00 | Operator | Fresh G5 baseline; 5 h slot covers 4.753 h projection + completion copy |
| Sep 13 13:00 | Sep 13 14:00 | Operator | 30 min verification/export then 30 min backup; eligible endpoint required |
| Sep 13 14:00 | Sep 14 14:00 | Operator | One full evaluation, provisional 24 h; verified export required |
| Sep 14 14:00 | Sep 15 14:00 | Operator | 24 h downtime/repair contingency; no overlapping machine work |
| Sep 15 14:00 | Sep 15 16:00 | Operator | Replay/restart contingency and result reconciliation; 2 h slot |
| Sep 15 16:00 | Sep 17 00:00 | Operator | 32 h unallocated internal margin; not a measured calibration budget |
| Sep 17 | Sep 18 target | Both humans | Release review, public assets and submission preparation |

The scheduled slots total 56 h; their slack above the 54.057 h scenario covers rounding
and reconciliation. If prerequisites miss Sep 13 08:00, rebase the entire sequential
calendar. Fit can only be justified after availability, evaluation measurement and
any calibration cost, backup/restore timing and approvals are resolved within the
internal boundary. Deadline clarification is separately required for a firm submission
claim. No blanket claim that all work fits is made.

## Historical cost custody

Keep section 3's two outer wrapper invocations (2562.060 s), section 4's successful
controller (543.071 s), and failed pre-training controller (3.036 s): **3108.167 s
known nonoverlapping subset**. Section 4's 389.783 training-command seconds and
15.969 archived optimizer seconds are nested, not extra additions. The archived
1,048,576 tokens remain costs outside canonical progress. These controller scopes
exclude final hashing/imports/review; earlier experiments, smokes, G0–G3 work and
verification are also outside this subtotal. Complete campaign wall and active GPU
hours remain null. Future accounting must reconcile all successful, failed,
interrupted and repeated invocation receipts without summing nested timers.

## Missing inputs and disposition

| Input / requirement | Status | Owner and next action |
|---|---|---|
| Fixed horizon arithmetic; measured profile applicability | PASS | Root rechecked recipe, raw metrics and 172 source hashes |
| Full CUDA/BF16 evaluation p90 and exact calibration cost | NOT_RUN | Operator authorizes separate eligible-artifact calibration; recompute reserve |
| Measured campaign fit | BLOCKED | Operator supplies runtime and confirms calendar availability |
| Machine availability | BLOCKED | User confirms uninterrupted windows; then rebase if needed |
| Submission cutoff conflict / scoring settings | BLOCKED | Operator obtains organizer clarification; preserve immutable protocols |
| Backup destination, custody and restore timing | NOT_RUN | Section 7 records policy and actual verification evidence |
| Two-person freeze approval | BLOCKED | Two distinct humans approve the same section-7 bundle digest |
| G4 amended / original gate, G5 baseline, G6 release | NOT_RUN | Complete applicable evidence/approvals; no execution in section 6 |

Optional 3–5B work and at most one comparison remain excluded from this candidate
count/calendar until baseline export and full evaluation are secured and a separate
measured budget supports them. A new horizon requires a new declared run identity;
never extend this fully decayed baseline silently.

Verification: calculation totals, source hashes, event/write counts and documentation
links checked. Production code and frozen contracts unchanged; prior source-bound
826-test Docker/Windows, lint and build evidence remains applicable. Next safe step:
`Get-Content docs/g4/07-freeze.md`.
