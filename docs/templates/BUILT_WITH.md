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

Pinned in `configs/data/sources_v2.yaml`. These rows are facts, not placeholders — copy them
into the Devpost Built With field verbatim.

| Dataset | Revision | License | Role |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0 | 70% stable + reserved science/edu-decile |
| mlfoundations/dclm-baseline-1.0 | `a3b142c183aebe5af344955ae20836eb34dcf69b` | CC BY 4.0 | 20% stable |
| open-web-math/open-web-math | `fde8ef8de2300f5e778f56261843dab89f230815` | ODC-By 1.0 | 7% stable + reserved math prose |
| storytracer/US-PD-Books | `01f85b67ba15cc3275e36d84ff51b23c90ce190a` | CC0 1.0 | 3% stable narrative |
| wikimedia/wikipedia | `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` | CC BY-SA 3.0, GFDL | reserved pool |
| izumi-lab/open-text-books | `1245fefd628d37483366b8e707fdc5650fd3c48e` | CC BY-SA 4.0 | reserved pool |

FineWeb-Edu and OpenWebMath derive from Common Crawl; also observe the
[Common Crawl ToU](https://commoncrawl.org/terms-of-use/). The Wikipedia portion is share-alike.

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
