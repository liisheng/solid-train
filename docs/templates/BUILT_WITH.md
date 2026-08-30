# Built With (template)

> **Template.** Plan Section 16 requires frameworks, libraries, APIs, datasets, AI tools, **and
> both GPUs**. List what was actually used. Omitting a used tool is a disclosure failure; the
> GPUs are the item most often forgotten.

## Frameworks and libraries

Exact versions come from `constraints/verified-py311-windows.txt` and
`scripts/check_environment.py`, never from memory.

| Component | Version | Purpose |
|---|---|---|
| Python | `TBD` | runtime |
| PyTorch | `TBD` | model, training, SDPA attention |
| NumPy | `TBD` | token storage, deterministic bootstrap |
| tokenizers / Hugging Face `datasets` | `TBD` | tokenizer training, streaming corpus acquisition |
| `lm-evaluation-harness` | `TBD` (commit `PENDING_PIN`) | required benchmark evaluation |
| pytest / Hypothesis | `TBD` | property and contract tests (test extra only) |

## Datasets

| Dataset | Revision | License | Role |
|---|---|---|---|
| `TBD` | `PENDING_PIN` | `TBD` | `TBD` |

Full detail in `docs/templates/DATA_CARD.md`.

## Hardware — list both

| GPU | VRAM | Lane |
|---|---|---|
| `TBD` | `TBD` | mainline: capacity tests, LR safety, continuous stable lineage, canonical archive |
| `TBD` | `TBD` | data and branches: preprocessing, tokenizer, packing, proxies, A/B/C, evaluation |

Host CPU, system RAM, and disk: `TBD` — from the Section 9.1 hardware inventory.

## AI tools

See `docs/templates/AI_ASSISTANCE_DISCLOSURE.md`. Every tool listed there is also listed here.

## APIs and hosted services

| Service | Purpose | Touches the submitted model? |
|---|---|---|
| `TBD` | `TBD` | must be **no** |

No hosted model performs inference for the submitted model, labels or rewrites training text,
or is a required evaluation dependency.
