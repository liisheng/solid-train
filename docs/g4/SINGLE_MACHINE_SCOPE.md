# Single-machine campaign decision

Recorded 2026-09-12T13:54:00+08:00. The user explicitly chose to do the remaining
work on this RTX 4070 SUPER machine and authorized updating the plan.
The versioned authority is [g4_scope_v2.yaml](../../configs/operations/g4_scope_v2.yaml).
It supersedes the active reduced campaign's takeover requirement, including the
takeover preservation clause in submission_scope_v2; frozen originals remain intact.

**Section 5 is not applicable under the amended scope. Section 6 is next.**
No target-machine transfer, profile or resume is required. The replacement is
verified local recovery, already supported by [section 4](RECOVERY_EVIDENCE.md).
The sustained source-machine profile is in [section 3](PROFILE_EVIDENCE.md).
Neither cross-machine takeover nor the original canonical G4 is claimed passed.

Section 6 must budget training, evaluation and export sequentially on this machine,
including local recovery and downtime reserves. Full evaluation runtime remains
unmeasured until actual evidence exists. If this machine fails, the campaign waits
for restoration; changing machines later requires a scope/compatibility review.

Section 7 must include the amendment's exact digest in the freeze bundle, assess its
four requirements, and record the checkpoint backup destination, hashes and restore
procedure. Storage outside this machine is recommended; no backup has been performed
or verified by this decision. Two-person freeze approval and release approvals remain.
Use G4_PASS_UNDER_AMENDED_SCOPE only after all applicable evidence and approvals pass.
The existing operations validator still checks the original frozen contract; do not
feed it invented takeover evidence or describe its result as the amended assessment.

This update launches no training, scoring, backup, transfer or publication. Existing
source/configuration measurements remain historical evidence with unchanged bytes;
the later freeze must bind the added scope record and check evidence applicability.
