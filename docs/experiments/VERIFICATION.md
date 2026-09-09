# Implementation verification — 2026-09-09

Status: **package verified; GPU experiments not run**. This is preparation evidence,
not a winning recipe, measured GPU readiness, or a G3/G4 gate pass.
Verification was performed on `codex/pre-campaign-experiments`, based on `4e27266`.
Publication amendment 2026-09-09: the user authorized committing and pushing the
verified package and required G4 tooling on `codex/g3.5-pre-campaign-experiments`.
This changes documentation and publication state, not the reviewed implementation.

Astra planned and supplied the implementation specification. Luna at medium
reasoning authored the scripts, validation extension, tests and initial handoff.
Root integrated and repaired the combined implementation. Sol and Terra, both at
medium reasoning, independently reviewed it and approved the final source. Root
matched their seven reviewed file hashes against the final checkout.

## Checks completed

| Check | Result |
|---|---|
| Full CPU container suite, current source and tests mounted read-only | 795 passed in 159.58 seconds |
| Sol independent focused experiment suite | 39 passed in 14.93 seconds |
| Final Dockerfile build | PASS |
| Final-image Ruff, compilation and dependency consistency | PASS |
| Final image versus tested checkout | Source and test fingerprints match |
| Installed wheel versus source | All 36 Python modules match |
| Real production input metadata, tokenizer and shard payload custody | PASS |
| Real CPU prepare and trainer argument/identity checks | S0, SLR, SMIX, C0 and both possible C1 treatments PASS |
| Continuous versus interrupted/resumed actual tiny CPU trainer | Exact model/optimizer/counter/cursor and endpoint slice evidence checks PASS |
| Whitespace/diff check | PASS; Git reports expected LF/CRLF conversion notices |

The real argument checks used the production configuration, manifests, schedules,
batch and horizon, but did not allocate/train a production model. The BF16 policy
was simulated solely to exercise the argument checker, not measured on a GPU.
The self-contained CPU trainer fixture is a separate tiny model; its result does
not imply production throughput, memory fit or benchmark quality.

Both schedules contain 97,792 references. Control quotas are educational 68,454,
general 19,558, math 6,846, narrative 2,934. Candidate quotas are educational
83,122, general 4,890, math 6,846, narrative 2,934. Root and Sol verified exact
math/narrative reference multiset preservation, beyond merely matching counts.

## Evidence locations

Local ignored artifacts must be transferred separately from source control:

- `runs/pre_campaign/v2/bundle.json`: final prepared bundle; SHA-256
  `95f1a249374e452f32549824d5624df0ee57f8d57c140d5a698763707d26549b`.
- `runs/pre_campaign/v2/cpu_preflight/report.json`: six parsed trainer checks,
  91.437 seconds of CPU preparation/checking; no GPU cost or speed claim.
- `runs/pre_campaign/v2/cpu_preflight/source_verification.json`: full source
  fingerprint, final reviewed hashes, image/test comparison and installed-wheel check.
- `runs/pre_campaign-tests-final.log`: full suite output.
- `runs/pre_campaign-build-verified.log`: final Docker build output.
- `runs/pre_campaign-prepare-v2-final.log`: final CPU preparation output.

Final image: `tinybench-lm:verify`, ID
`sha256:00e4a4ea15a1ca1dc1068acc4aca1c7a44a532f03f98cd7042c49c802ffdf570`.
Earlier preparation bundles were preserved as `v2-before-final-review` and
`v2-before-portability-fix`; they are stale, contain no launched jobs, and must not
be used. The final bundle has no `jobs` or `smoke` execution directories.

## Final review hashes

| File | SHA-256 |
|---|---|
| train.py | a33149b18bb5e78340def20d1872535dbac9111b06670f14eee2dccb1fd29555 |
| scripts/run_experiment.py | 8dd530ccf0c7265fe6e25909d870723d6c26983ab32021e7292888dc6b643e0b |
| scripts/analyze_experiments.py | 639ff8a1ff1bf2d80a5b6b4fc6ba52bb133cae92f3e0baf9b2c1b05e15e48c0b |
| src/tinybench_lm/experiments.py | 8216508e88dcca9d7af91527e0c9eb07fd080bcc1a6909191bd2da413897514e |
| src/tinybench_lm/experiment_budget.py | 8f28f44f5ef3bc8c2817bb945131524aaba617cedbf3d580daaecbcbe50670c6 |
| configs/campaign/pre_campaign_v2.json | 31d5093618e366e4f8b393a61c692459450f79db0aaeb7a352aec6b82df0a8d4 |
| configs/campaign/submission_scope_v2.yaml | 3e8cd561080b107a90bd7ff7bcefdfffa76e8f00238d4d03aa1ab938de2c5f7a |

## Next operator task

Use OPERATOR_GUIDE.md and LLM_HANDOFF.md. Establish actual hardware/environment,
available time and remaining allowance, then run each authorized smoke and job.
Record a complete result or explicit incomplete closure. A selected-settings
handoff still needs successor baseline integration and G4 sections 3–7 before G5.
No production experiment, main training, submission scoring, commit, push or model
publication occurred during this implementation task. Existing unrelated files
and the prior G4 tooling were preserved.
