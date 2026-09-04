from __future__ import annotations

from scripts.fetch_decontamination_inputs import extract_texts


def test_extract_texts_follows_declared_nested_fields_and_deduplicates() -> None:
    row = {
        "question": " What is two plus two? ",
        "choices": {"text": ["three", "four", "four"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    assert extract_texts(row, ("question", "choices.text")) == (
        "What is two plus two?",
        "three",
        "four",
    )


def test_extract_texts_ignores_unlisted_metadata_and_empty_values() -> None:
    row = {"ctx": "A useful context", "id": "secret-id", "endings": ["", " conclusion "]}
    assert extract_texts(row, ("ctx", "endings")) == ("A useful context", "conclusion")
