"""Explicit dataset bindings for provisional evaluation; never rely on task aliases."""

from __future__ import annotations

from typing import Any, Mapping

from .evaluation_protocol import task_entry


def nonempty_wikitext_rows(dataset):
    """Keep raw nonempty paragraphs; preserve dataset order and text bytes."""
    return dataset.filter(lambda row: bool(row["text"].strip()))


def raw_wikitext_metrics(doc, results):
    """Word/byte denominators for the explicitly declared raw-paragraph protocol."""
    (loglikelihood,) = results
    text = doc["text"]
    words = len(text.split())
    byte_count = len(text.encode("utf-8"))
    if not words or not byte_count:
        raise ValueError("Empty WikiText rows must be removed before scoring")
    return {
        "word_perplexity": (loglikelihood, words),
        "byte_perplexity": (loglikelihood, byte_count),
        "bits_per_byte": (loglikelihood, byte_count),
    }


def resolve_harness_tasks(names: list[str], protocol: Mapping[str, Any]) -> list[Any]:
    """Bind WikiText-103 to the pinned raw dataset instead of harness WikiText-2."""
    resolved: list[Any] = []
    for name in names:
        if name == "wikitext":
            raise ValueError("The legacy wikitext alias loads WikiText-2. Use evaluation_provisional_v2 and wikitext103.")
        if name != "wikitext103":
            resolved.append(name)
            continue
        entry = task_entry(protocol, name)
        binding = entry["dataset_binding"]
        if binding["repo"] != "Salesforce/wikitext" or binding["config"] != "wikitext-103-raw-v1":
            raise ValueError("WikiText-103 dataset identity mismatch")
        resolved.append({
            "task": name,
            "dataset_path": binding["repo"],
            "dataset_name": binding["config"],
            "dataset_kwargs": {"revision": entry["dataset_revision"]},
            "test_split": "test",
            "output_type": "loglikelihood_rolling",
            "process_docs": nonempty_wikitext_rows,
            "doc_to_text": "",
            "doc_to_target": "{{text}}",
            "process_results": raw_wikitext_metrics,
            "metric_list": [{"metric": key, "higher_is_better": False,
                             "aggregation": "bits_per_byte" if key == "bits_per_byte" else "weighted_perplexity"}
                            for key in entry["metric_keys"]],
            "metadata": {"version": 1, "scoring": "raw_nonempty_paragraphs_word_and_byte_denominators"},
        })
    return resolved
