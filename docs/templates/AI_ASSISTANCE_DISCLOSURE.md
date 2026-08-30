# AI assistance disclosure (template)

> **Template.** Plan Section 11.2 requires disclosing the *exact* assisted project areas. Fill
> this in from what actually happened, not from what was permitted. Under-disclosure is a
> compliance failure; over-disclosure costs nothing.

## Permitted uses, and whether each was used

| Permitted use (Plan 11.2) | Used? | Tool | Where |
|---|---|---|---|
| Code and config review | `TBD` | `TBD` | `TBD` |
| Failure-log summarization | `TBD` | `TBD` | `TBD` |
| Literature and dataset-card summarization | `TBD` | `TBD` | `TBD` |
| Documentation and video-script drafting | `TBD` | `TBD` | `TBD` |
| Comparing manifests for unintended differences | `TBD` | `TBD` | `TBD` |

## Forbidden uses — all must remain "no"

| Forbidden use (Plan 11.2) | Occurred? | Evidence |
|---|---|---|
| Submitted-model inference | no | `src/tinybench_lm/eligibility.py` |
| Generated or rewritten training text | no | `configs/data/sources_v1.yaml` |
| Teacher labels, logits, probabilities, or rankings | no | `src/tinybench_lm/eligibility.py` |
| Benchmark-targeted generation | no | `configs/data/decontam_v1.yaml` |
| Required-evaluation dependency | no | `configs/evaluation/evaluation_provisional_v1.yaml` |

Any "yes" in this table voids eligibility. The automated eligibility scan
(`tests/test_eligibility.py`) checks the repository for these patterns; it cannot observe what
happened outside the repository, so the human attestation below is required as well.

## Assistant configuration

- Primary coding assistant: `TBD`
- Reviewer assistant: `TBD`
- API keys supplied via environment variables only, never committed: `TBD`
- Concurrent agents editing this repository were avoided: `TBD`

## Attestation

Each teammate confirms this disclosure is complete and accurate for their own work.

| Teammate | Date | Confirmed |
|---|---|---|
| `BLOCKED` | `BLOCKED` | `BLOCKED` |
| `BLOCKED` | `BLOCKED` | `BLOCKED` |
