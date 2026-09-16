# TinyBench-LM v1.0.0 model card

- Model type: English causal text-completion Transformer, trained from scratch.
- Parameters: 49,658,368 unique trainable parameters; tied input/output embedding.
- Context: 1,024 tokens. Vocabulary: 12,288-token byte-level BPE.
- Architecture: 14 layers, width 512, 8 query / 4 key-value heads, SwiGLU width 1,504; RoPE and RMSNorm.
- Training: 1,000,079,360 loss tokens, 3,815 updates, seed 1337, BF16, RTX 4070 SUPER.
- Data mixture: FineWeb-Edu 70%, FineWeb 20%, OpenWebMath 7%, Gutenberg English 3%.
- Intended use: research, reproducibility, parameter-constrained language modeling, and GIBC V2 evaluation.
- Limitations: repetition, factual errors, weak commonsense performance, no instruction tuning or established conversational usefulness.
- Evaluation: all scores provisional; see the README table and release evidence archive.
- Provenance: initial-weight record and final export SHA-256 are published; no pretrained initialization or distillation.
- Release files: baseline_export.pt, tokenizer.json, final_49m.json, evaluation-evidence.zip, SHA256SUMS.txt.
- Licensing: no new model license is asserted by this release. Dataset licenses and source credits are recorded in the README; public download alone does not establish a new reuse license.

The original export format is loaded by this repository's `generate.py` and harness
adapter. This is a custom PyTorch model, not a Transformers `AutoModel` package.
Use the release checkout and documented commands. Do not substitute another tokenizer.
