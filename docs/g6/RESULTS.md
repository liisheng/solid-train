# G6 provisional evaluation results

2026-09-13T01:23+08:00. Full evaluation completed and the saved bundle was
independently reopened and verified. **PROVISIONAL_NOT_OFFICIAL**. G6 release is
still blocked on official settings/provenance, public release assets and human approvals.

| Task | Examples | Accuracy | Normalized accuracy |
| --- | ---: | ---: | ---: |
| HellaSwag | 10,042 | 27.30% | 27.52% |
| ARC-Easy | 2,376 | 41.33% | 37.50% |
| PIQA | 1,838 | 58.65% | 58.11% |
| WinoGrande | 1,267 | 49.96% | Not emitted |

WikiText-103: all 2,891 nonempty test paragraphs; word perplexity 235.1593,
byte perplexity 2.7811, bits per byte 1.47566. This word-based perplexity is not
directly comparable to G5's token-based development perplexity 26.7664 on a different
dataset. Raw results retain every emitted metric and standard error.

These results establish complete provisional scoring, not conversational usefulness.
WinoGrande is effectively at the 50% binary-choice chance baseline. No benchmark scores
may be used to select training settings under the frozen evaluation contract.

## Verification and environment

All five trusted revisions/splits/counts and each document ID verified, with no limit
or duplicate/missing examples. Controller verification and a subsequent root
`postverify.py` invocation passed. The source snapshot and export hashes were rechecked
by the controller after scoring. Export/tokenizer hashes are the G5 identities listed
in [the handoff](README.md). Model: 49,658,368 parameters; CUDA/BF16 on RTX 4070 SUPER,
batch 16, seed 1234, zero-shot, bootstrap 1,000.

The fresh environment reports Python 3.12.6 and torch 2.5.1+cu124. The constraints
file's historical Python 3.11.4 header is not the measured interpreter; the project
supports Python >=3.11 and the fresh environment verifier passed. Preserve this fact
in comparisons rather than calling this exact interpreter reproduction.

Setup's final environment check initially failed because an installed package derived
`venv/Lib/pyproject.toml` instead of the snapshot path. Dependency installs and pip
check all passed. The controller reran with snapshot `PYTHONPATH`; environment check
passed before scoring. The original failure receipt is retained, superseded for
readiness by `fresh-environment.json`; this was not a dependency or scoring failure.

## Measured time and accounting

- Fresh setup controller: 223.372 s, including its failed path lookup.
- Evaluation process: 192.305 s (3 min 12.3 s), including loading and evidence output.
- Harness scoring inside that process: 99.558 s; do not add it again.
- Corrected environment check: 1.926 s; controller postverification: 3.234 s.
- Evaluation controller: 233.133 s, including waiting for setup. Do not add its full
  duration to setup because the waits overlap.
- Nonoverlapping setup plus corrected check, evaluation and postverification:
  420.838 s (~7 min 1 s). This excludes Docker/code verification and later root review.
- G5 launch/completion controllers: 15,952.534 s. Adding the above gives 16,373.372 s
  (~4 h 32 min 53 s) for these named G5/G6 phases only, not total project compute or active
  GPU time. Earlier experiments/G3/G4 and code verification remain outside that subtotal.

One measured run replaces the prior unknown runtime for this invocation only; it does
not establish p90 runtime. The 24 h provisional allowance was not consumed or enforced
as a measured forecast.

Evidence under `runs/verification/g6/20260913-01`: `receipt.json`,
`fresh-environment.json`, `results.json`, `bundle/`, `postverify.stdout.log`,
`root-postverify.json`, source hashes and setup logs. These are local ignored artifacts.
No benchmark rerun, training, model modification, publication, commit or push occurred
during root review. Release packaging/public accessibility and required human approval
remain pending; no G6 PASS is claimed.
