# Submission package (template)

> **Template.** Nothing here is checked off in advance. A tick means the named evidence exists
> and someone verified it; `BLOCKED` means an external party must act; `NOT_RUN` means the
> artifact does not exist yet. The machine-readable source of truth is
> `configs/release/evidence_matrix_v1.yaml`, verified by `src/tinybench_lm/release.py`.

## Status legend

| Status | Meaning |
|---|---|
| `PASS` | a named verifier ran and the artifact exists at the named path |
| `FAIL` | a verifier ran and did not pass |
| `BLOCKED` | an organizer, teammate, or host must act first — owner and next action recorded |
| `NOT_RUN` | the verifier exists but has not run, or the artifact does not exist yet |
| `TBD` | a value only a future measurement can supply |

## Eligibility and registration — `BLOCKED`

Personal eligibility is an attestation no repository check can produce. Never assert it on a
teammate's behalf.

- [ ] Teammate 1 confirms eligibility and Devpost registration — `BLOCKED`
- [ ] Teammate 2 confirms eligibility and Devpost registration — `BLOCKED`
- [ ] One team, one track, real names added — `BLOCKED`

## Public artifacts — `BLOCKED` until published and checked logged out

- [ ] Source repository public and unrestricted — `BLOCKED`
- [ ] Model weights accessible without private credentials — `BLOCKED`
- [ ] Weights mirrored so no single remote host is the only verified copy — `NOT_RUN`
- [ ] Logged-out access check — `NOT_RUN`
- [ ] Fresh-clone check — `NOT_RUN`

Record the exact URL and the date and time each check was run. A link that works while signed
in has not been verified.

| Artifact | URL | Logged-out check | Date |
|---|---|---|---|
| Repository | `BLOCKED` | `NOT_RUN` | — |
| Weights | `BLOCKED` | `NOT_RUN` | — |
| Video | `NOT_RUN` | `NOT_RUN` | — |

## Screenshots — at least three, target five

| # | Subject | Path | Status |
|---|---|---|---|
| 1 | Training loss / tokens / LR / branch curve | `NOT_RUN` | `NOT_RUN` |
| 2 | A/B/C ablation with confidence intervals | `NOT_RUN` | `NOT_RUN` |
| 3 | Parameter counter beside the model config | `NOT_RUN` | `NOT_RUN` |
| 4 | Evaluation command and raw output | `NOT_RUN` | `NOT_RUN` |
| 5 | Data/artifact lineage or two-machine workflow | `NOT_RUN` | `NOT_RUN` |

Every screenshot shows real output. A mock-up is never acceptable.

## Video — approximately four minutes, English audio or subtitles, public or unlisted

- [ ] 1. Constraint and project positioning — `NOT_RUN`
- [ ] 2. Architecture and live parameter proof — `NOT_RUN`
- [ ] 3. Corpus, decontamination, reserved reasoning pool — `NOT_RUN`
- [ ] 4. Three-arm experiment and what B versus C isolates — `NOT_RUN`
- [ ] 5. Final benchmark, perplexity, throughput, memory — `NOT_RUN`
- [ ] 6. Bounded live CPU smoke evaluation plus timestamped full results — `NOT_RUN`
- [ ] 7. Limitations, reproducibility, credits, AI disclosure — `NOT_RUN`

Never private. Do not present a 25-second clip as a complete evaluation suite: either show a
bounded smoke run or start the full command and show a timestamped completed result.

## Results — provisional until organizer answers arrive

- [ ] Required four harness tasks reported with exact protocol — `NOT_RUN`
- [ ] WikiText-103 perplexity with the organizer-specified slice — `BLOCKED`
- [ ] Raw JSON results committed — `NOT_RUN`
- [ ] Secondary reasoning tasks labeled non-official — `NOT_RUN`

## Efficiency figures — measured, never targeted

- [ ] Hardware identity for both GPUs — `TBD`
- [ ] Total wall time and active GPU-hours — `TBD`
- [ ] Approximate compute (6ND estimate, labeled approximate) — `TBD`
- [ ] Median / p10 / p90 throughput per machine — `TBD`
- [ ] Peak VRAM and system RAM per machine — `TBD`
- [ ] Accepted, consumed tokens, effective passes, tokens/parameter — `TBD`

20B tokens and 402.8 tokens/parameter remain stretch targets until reached.

## Hashes and approval

- [ ] Final commit / tag recorded — `NOT_RUN`
- [ ] Final artifact hash recorded — `NOT_RUN`
- [ ] Rollback artifact hash recorded — `NOT_RUN`
- [ ] Both teammates approve the exact release hash — `BLOCKED`

| Item | Value | Approved by | Date |
|---|---|---|---|
| Release hash | `NOT_RUN` | `BLOCKED` | — |
| Rollback hash | `NOT_RUN` | `BLOCKED` | — |

## Judge verification tiers

| Tier | What a judge does | Prepared? |
|---|---|---|
| 0 | Inspect committed raw JSON results | `NOT_RUN` |
| 1 | CPU load and small evaluation, measured and documented | `NOT_RUN` |
| 2 | Tiny data → train → eval reproduction on CPU | `NOT_RUN` |
| 3 | Full training reproduction with manifests and appropriate GPU/storage | `NOT_RUN` |

## Monitoring window

- [ ] Both teammates monitor Devpost and email September 22-27 — `BLOCKED`
- [ ] Rota agreed before the deadline — `BLOCKED`

Respond to organizer contact within hours.

## Stop conditions

Do not add another architecture, data family, optimization method, dashboard, frontend, or
hosted-model workflow unless measured evidence creates a specific need and the critical
schedule is ahead.

If branches fail: ship the best valid ordinary-decay or stable fallback (`fallback-v1`),
preferring arm A when no experimental effect is supported. Report null or incomplete
experiments honestly. **Never release an undecayed peak-LR mainline checkpoint as the
fallback.**
