# G4 section7 freeze review

Recorded 2026-09-12. **Section7 package complete; amended G4 BLOCKED.** Original
canonical G4 remains NOT_RUN. This is an evidence assessment and paraphrase of
the governing contracts, not an amendment or launch authorization.

## Bundle and scope

`FREEZE_BUNDLE.json` is the stable payload; `FREEZE_BUNDLE.json.sha256` holds its
single SHA256 over exact UTF-8 file bytes. It binds source, inputs, evidence and
the G5 procedure. All file digests inside it use raw bytes, independently of the
contracts' LF-normalized semantic hashes. In particular CONTROL v2 has raw digest
`3fe98a6f969ece2248ca654032a7e05de7e4f7404a52751ae82fa3270b860f24`
and normalized digest `dcd4d623a8f826f5e007ed33eea4bf2b653414c2088f4354a9b7e6e69a8a3f12`.

Authority: submission_scope_v1/v2, selected baseline_reduced_v2 and g4_scope_v2.
The latter's raw digest is
`9a7ff182683379afb6f04ac1d8cfbce74cd9482dd5c20e2a023d4f3614866a1a`.
Frozen originals remain unchanged, including pending fields and historical statuses.
Takeover is excluded only for this single-machine scope. No original full-scale
G0–G6 pass is inferred. Release still requires its own human approvals.

## Requirement-to-evidence matrix

| Requirement | Disposition | Evidence / remaining owner action |
|---|---|---|
| Two-person freeze approval | BLOCKED | Zero actual receipts. Operator and second distinct human review the same final bundle digest; AI reviews do not count. |
| Sustained real-shard throughput | PASS | PROFILE_EVIDENCE; 552 included updates/2243.512 optimizer seconds,64,498.66tokens/s,31.827% sampled device headroom;17 postverification checks true. |
| Local recovery | PASS | RECOVERY_EVIDENCE;20 checks true, exact CUDA BF16 eight versus four+resume and rollback; corruption and early export rejected. Abrupt process kill and FP16 scaler recovery are not claimed. |
| Original cross-machine takeover | NOT_APPLICABLE_UNDER_AMENDED_SCOPE | g4_scope_v2 replaces it with local recovery; original takeover remains NOT_RUN. |
| No failed correctness gate bypassed | PASS in inspected implementation scope |172 source files unchanged;826 Docker and826 Windows tests, build/lint/dependencies retained; failed attempts preserved. External readiness gaps below remain blockers. |
| Complete freeze components | BLOCKED | Component map in bundle covers all13 preregistration components plus backup. Actual destination/custodian/cadence and restore custody are unresolved. Operator supplies policy and evidence. |
| Fixed fresh horizon | PASS |3815updates/1,000,079,360tokens/WSD38/3395/382; successful section7 plan; fresh G5 directory absent. No final checkpoint hash exists. |
| Machine availability | CONFIRMED_BY_USER | User commits to keeping this computer on as long as needed; sequential local policy retained. Not proof of uptime or runtime fit. |
| Measured budget and calendar fit | BLOCKED | BASELINE_BUDGET arithmetic and14 input hashes pass. Full CUDA/BF16 evaluation p90 NOT_RUN;24h allowance provisional. Operator resolves measurement/eligible-artifact dependency and calibration cost, or obtains an explicit accepted scope disposition; section7 does not waive measured-fit requirements. |
| Official evaluation / submission readiness | BLOCKED, downstream | Effective loader binding retained; official harness commit provenance and organizer settings remain unresolved in EVALUATION_EVIDENCE. Historical deadline conflict remains unresolved. Operator obtains answers before official result/release claims. |

## Source applicability and accounting

Section7 rehashed462 unique files:172 measured source entries,15 production input
entries,43 profile evidence entries,90 recovery entries,13 integration evidence
entries,14 budget source entries and133 payload entries (groups overlap).
All match. Existing profile/recovery check trees and budget arithmetic pass.
Evidence: `runs/verification/g4/section-07/20260912-freeze-01/custody.json`.
No trainer/config/reader/evaluation implementation changed; the previously bound
826-test suites, lint and builds remain applicable. New work is a documentary freeze
and raw-file verification; no GPU remeasurement or full test rerun is warranted.
This does not certify future machine load; refresh environment and resources at G5.

The successful read-only launch plan exits0, retaining runner
`baseline-f23398e32dc3100e`, with no resume/engineering stop. Initial plan attempt
exited1 on sandbox manifest access before training; stderr remains alongside the
successful access-enabled retry. Section7 performs no training, scoring or backup.

The bundle retains successful/failed section4 receipts and section3 invocation history.
Section6's3108.167s known nonoverlapping historical subset is unchanged; nested
child/optimizer times are not added again. Archived1,048,576tokens/15.969optimizer
seconds remain charged outside canonical progress. Section7 verification is additional
engineering work, not training speed evidence; its custody timer is recorded separately.
Complete campaign wall and active GPU hours remain unknown.

## Superseded components and approval procedure

Original proxy/three-arm parent selection and branch calendar are superseded by
submission_scope_v1/v2. Bind the actual CONTROL decision and fresh seed; there is no
proxy parent checkpoint. The active calendar is one baseline, export, evaluation and
local contingency; no optional comparison/extension is scheduled. Fallback is local
verified recovery and waiting for machine restoration. There is no eligible released
fallback model yet. G5_HANDOFF binds the proposed backup/restore procedure and clearly
marks its missing destination, custodian and proof.

Review the package now, but close material gaps and regenerate it before launch
approval. Keep actual approval receipts outside the payload, under
`docs/g4/approvals/` when supplied. Each receipt must contain bundle SHA256, distinct
human identity, ISO timestamp, explicit approval/rejection, scope
`selected_control_reduced_baseline_v2_single_machine`, and the original approval
statement/provenance. No receipts have been fabricated. Approval of a blocked review
alone does not waive its blockers. Any payload change needs two new matching receipts.
This review and coordination files are excluded from the payload to avoid hash cycles;
the machine-readable matrix and G5_HANDOFF are inside it.

Safe inspection:

```powershell
Get-FileHash docs/g4/FREEZE_BUNDLE.json -Algorithm SHA256
Get-Content docs/g4/FREEZE_BUNDLE.json.sha256
Get-Content docs/g4/G5_HANDOFF.md
```

Next owner actions: operator names backup destination/custodian/cadence and supplies
restore custody; operator resolves budget disposition; two humans then approve the
updated exact bundle. G5 is separately requested after readiness. No long-running
script remains from section7.
