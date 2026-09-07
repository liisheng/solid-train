from pathlib import Path

import pytest

from tinybench_lm.evaluation_protocol import load_evaluation_protocol
from tinybench_lm.evaluation_tasks import raw_wikitext_metrics, resolve_harness_tasks


def test_wikitext103_has_explicit_pinned_identity():
    protocol = load_evaluation_protocol(Path("configs/evaluation/evaluation_provisional_v2.yaml"))
    tasks = resolve_harness_tasks(["hellaswag", "wikitext103"], protocol)
    assert tasks[0] == "hellaswag"
    assert tasks[1]["dataset_path"] == "Salesforce/wikitext"
    assert tasks[1]["dataset_name"] == "wikitext-103-raw-v1"
    assert tasks[1]["dataset_kwargs"]["revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert tasks[1]["test_split"] == "test"


def test_legacy_alias_cannot_be_misreported_as_103():
    with pytest.raises(ValueError, match="WikiText-2"):
        resolve_harness_tasks(["wikitext"], load_evaluation_protocol())


def test_raw_paragraph_denominators_count_words_and_utf8_bytes():
    result = raw_wikitext_metrics({"text": "  café\n moon  "}, [-12.0])
    assert result["word_perplexity"] == (-12.0, 2)
    assert result["byte_perplexity"] == (-12.0, 15)
    with pytest.raises(ValueError, match="Empty"):
        raw_wikitext_metrics({"text": " \n"}, [-1.0])
