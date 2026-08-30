"""PILOT ONLY: bounded corpus download, small tokenizer, and flat train/validation packing.

This is the pilot smoke-test path. It is deliberately preserved so the existing bounded
end-to-end checks keep working, and it is NOT the final tokenizer or the final data
contract:

- It trains a small pilot vocabulary (8,192 by default), not the frozen 12,288-ID tokenizer.
- It applies an NFKC normalizer, which is lossy and therefore not reversible.
- It writes one flat `train.bin`/`validation.bin` pair with no source tags, no reserved
  pool, and no near-duplicate cluster isolation.

This monolithic flat output contract is superseded. The final data contract is the
source-tagged uint16 shard layout with independent split manifests, frozen in
`configs/data/shards_v1.yaml`, implemented by `src/tinybench_lm/shards.py`, and produced by
`scripts/build_shards.py`.

The final tokenizer contract lives in `configs/data/tokenizer_v1.yaml` and is implemented by
`src/tinybench_lm/tokenizer.py`; build and verify it with `scripts/build_tokenizer.py`.
Never use this script's output for the final campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm


#: This module is pilot-only. The final tokenizer contract is tokenizer_v1.
PILOT_ONLY = True
PILOT_SCOPE = "PILOT_ONLY"
FINAL_TOKENIZER_CONTRACT = "configs/data/tokenizer_v1.yaml"
#: The final data output contract. This script's flat train/validation pair is superseded.
FINAL_SHARD_CONTRACT = "configs/data/shards_v1.yaml"
PILOT_BANNER = (
    "PILOT ONLY: this builds a small pilot tokenizer and a flat train/validation split for "
    f"smoke tests. It is not the final tokenizer contract ({FINAL_TOKENIZER_CONTRACT}); use "
    "scripts/build_tokenizer.py for that. The flat train.bin/validation.bin output is "
    f"superseded by the source-tagged shard contract ({FINAL_SHARD_CONTRACT}); use "
    "scripts/build_shards.py for that."
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def download_documents(args: argparse.Namespace, raw_path: Path) -> dict[str, int]:
    os.environ.setdefault("HF_HOME", str(args.cache_dir / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.cache_dir / "datasets"))
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.subset,
        split=args.split,
        streaming=True,
        cache_dir=str(args.cache_dir / "datasets"),
    )
    documents = 0
    utf8_bytes = 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as output:
        iterator = tqdm(dataset, desc="Downloading public text", unit="doc", mininterval=1.0)
        for row in iterator:
            value = row.get(args.text_column)
            if not isinstance(value, str):
                continue
            text = clean_text(value)
            if len(text) < args.min_chars:
                continue
            encoded_bytes = len(text.encode("utf-8"))
            output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            documents += 1
            utf8_bytes += encoded_bytes
            if documents % 250 == 0:
                iterator.set_postfix(docs=documents, mib=f"{utf8_bytes / 2**20:.1f}")
            if documents >= args.max_docs or utf8_bytes >= args.max_bytes:
                break
    return {"documents": documents, "utf8_bytes": utf8_bytes}


def iter_text(raw_path: Path):
    with raw_path.open("r", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)["text"]


def train_tokenizer(raw_path: Path, tokenizer_path: Path, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_text(raw_path), trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def is_validation_document(text: str, validation_fraction: float) -> bool:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") / (2**64 - 1)
    return bucket < validation_fraction


def pack_tokens(
    raw_path: Path,
    tokenizer: Tokenizer,
    output_dir: Path,
    validation_fraction: float,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    validation_path = output_dir / "validation.bin"
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise RuntimeError("Tokenizer is missing <|endoftext|>")
    counts = {"train_tokens": 0, "validation_tokens": 0}
    with train_path.open("wb") as train_file, validation_path.open("wb") as validation_file:
        for text in tqdm(iter_text(raw_path), desc="Packing token IDs", unit="doc"):
            token_ids = tokenizer.encode(text, add_special_tokens=False).ids + [eos_id]
            if max(token_ids, default=0) > 65_535:
                raise ValueError("Token ID does not fit uint16")
            values = np.asarray(token_ids, dtype="<u2")
            if is_validation_document(text, validation_fraction):
                values.tofile(validation_file)
                counts["validation_tokens"] += len(values)
            else:
                values.tofile(train_file)
                counts["train_tokens"] += len(values)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PILOT ONLY: download, tokenize, and pack a bounded public corpus for smoke tests",
        epilog=PILOT_BANNER,
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-docs", type=int, default=20_000)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--vocab-size", type=int, default=8_192)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pilot"))
    parser.add_argument("--reuse-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(PILOT_BANNER)
    args = parse_args()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    raw_path = args.output_dir.parent / "raw" / f"{args.output_dir.name}.jsonl"
    tokenizer_path = args.output_dir / "tokenizer.json"
    if args.reuse_raw and raw_path.exists():
        download_stats = {"documents": sum(1 for _ in iter_text(raw_path)), "utf8_bytes": raw_path.stat().st_size}
    else:
        download_stats = download_documents(args, raw_path)
    if download_stats["documents"] == 0:
        raise RuntimeError("No usable documents were downloaded")
    tokenizer = train_tokenizer(raw_path, tokenizer_path, args.vocab_size)
    token_stats = pack_tokens(raw_path, tokenizer, args.output_dir, args.validation_fraction)
    metadata = {
        "scope": PILOT_SCOPE,
        "final_tokenizer_contract": FINAL_TOKENIZER_CONTRACT,
        "final_shard_contract": FINAL_SHARD_CONTRACT,
        "output_contract": "flat train.bin/validation.bin (superseded by shards_v1)",
        "final_use_prohibited": True,
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_column": args.text_column,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "validation_fraction": args.validation_fraction,
        **download_stats,
        **token_stats,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
