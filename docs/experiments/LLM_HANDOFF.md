# Brief for the next LLM

Operate the approved pre-campaign v2 experiments on the assigned machine. First
read AGENTS.md, .agent/CONTINUITY.md, docs/STATUS.md, docs/ENVIRONMENT.md, and this
directory's README.md, IMPLEMENTATION_SPEC.md and OPERATOR_GUIDE.md. The operator
must authorize GPU execution before you use `--execute`. Package preparation and
tests are not production training evidence. Do not install host packages.

Use the reviewed checkout and its `.venv`. From the repository root the interpreter
is `.\.venv\Scripts\python.exe`. Execute the exact commands in OPERATOR_GUIDE.md.
Do not change settings, hashes, schedules, thresholds, budgets, or identities to
make a check pass. If a check fails, preserve the files and report the cause.

The RTX4070 lane runs S0, SLR, SMIX in that order; each job first needs its own
two-update smoke. Only when all three completed endpoints select a candidate does
the RTX3070 lane run C0 and C1, each with its own smoke. C1's single treatment is
recomputed from the screen evidence. A job trains 382 updates / 100,139,008 loss
tokens. The declared horizon stays 382 even during smoke and resume.

Both machines need identical reviewed source, tokenizer, production manifests,
shard payloads and schedules. Transfer the complete experiment bundle and all
checkpoint sidecars/results, then run `check` on the receiving machine. Inputs
outside the bundle must also be transferred using the paths in OPERATOR_GUIDE.md.
Each physical machine retains its own parent-directory lane ledger. Never copy
another machine's ledger over it or reset the allowance with a new checkout.

Before every GPU job, inspect remaining lane allowance and existing processes.
One GPU job may run at a time. The wrapper records actual GPU UUID/name/memory,
reserves time before training, requires verified smoke evidence, checks memory
headroom and estimates whether all remaining jobs fit. Estimates are conservative
admission checks, not sustained profiles or finish guarantees. Each lane has six
hours; together at most twelve. Stop by 2026-09-12T23:59:59+08:00.

Wait for each command's completion and inspect its exit status. Successful child
exit alone is insufficient: endpoint verification must also succeed. Preserve
identity, launch/exit/verification receipts, console logs, metrics, checkpoints,
sidecars, timing and ledger history. Resume only the same identity from a verified
latest checkpoint. For abandoned reservations, independently establish that the
child is dead before the documented recovery command; the reservation is fully
charged. A stale operation lock needs investigation, not ledger deletion.

Run screen/final analysis with the documented commands and distinct output files.
Only complete full-dev endpoints with all four real protected slices can select a
candidate. SLR/SMIX compare against S0; C1 compares against C0. Keep the completed
endpoint, not a best intermediate checkpoint. No validation_final or submission
benchmark may enter this selection loop. Do not claim statistical significance.
If work cannot finish, use the explicit incomplete closure with the documented
reason; INCOMPLETE_CONTROL is not a passed gate.

Report jobs completed/failed, selected settings and evidence hashes, actual hardware,
remaining/charged time on both lanes, unresolved issues and next steps. Integrate
the selected settings into a successor baseline recipe and run identity, then
complete G4 sections 3–7. Only after G4 may G5 train the fresh 1B-token baseline.
Experiment weights never initialize it. Do not commit, push or publish without
authorization. Target submission September 18, retaining September 19–21 reserve.
