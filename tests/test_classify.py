"""Tests for main.classify.

Unit tests use a fake model so they run in milliseconds and check only our own
logic. Tests marked "slow" load the real ESGify model and check its behaviour.
Run everything with `pytest`, or skip the slow ones with `pytest -m "not slow"`.
"""

import pytest
import torch

from main import classify, load_model


class FakeTokenizer:
    """Stands in for the real tokenizer: returns one dummy row per text."""

    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {"input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long)}


class FakeModel:
    """Stands in for ESGify: returns preset scores, one row per text."""

    id2label = {0: "A", 1: "B", 2: "C", 3: "D"}

    def __init__(self, scores):
        self.scores = torch.tensor(scores)

    def __call__(self, input_ids, attention_mask):
        return self.scores[: len(input_ids)]


# ---------- unit tests (fast, fake model) ----------

def test_returns_one_result_per_text():
    model = FakeModel([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    results = classify(["first", "second"], model, FakeTokenizer())
    assert len(results) == 2


def test_labels_sorted_by_score_descending():
    model = FakeModel([[0.1, 0.9, 0.5, 0.7]])
    [labels] = classify(["text"], model, FakeTokenizer())
    assert labels == [("B", 0.9), ("D", 0.7), ("C", 0.5)]


def test_each_text_gets_its_own_labels():
    model = FakeModel([[0.9, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9]])
    first, second = classify(["first", "second"], model, FakeTokenizer(), top_k=1)
    assert first == [("A", 0.9)]
    assert second == [("D", 0.9)]


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_top_k_controls_number_of_labels(top_k):
    model = FakeModel([[0.1, 0.2, 0.3, 0.4]])
    [labels] = classify(["text"], model, FakeTokenizer(), top_k=top_k)
    assert len(labels) == top_k


def test_scores_rounded_to_three_decimals():
    model = FakeModel([[0.123456, 0.0, 0.0, 0.0]])
    [labels] = classify(["text"], model, FakeTokenizer(), top_k=1)
    assert labels == [("A", 0.123)]


# ---------- integration tests (slow, real ESGify model) ----------

@pytest.fixture(scope="module")
def esgify():
    """Load the real model once and share it across all tests in this file."""
    return load_model()


@pytest.mark.slow
def test_discrimination_text_gets_discrimination_label(esgify):
    model, tokenizer = esgify
    [labels] = classify(["She faced employment rejection because of her gender."], model, tokenizer)
    assert labels[0][0] == "Discrimination"


@pytest.mark.slow
def test_pollution_text_gets_environmental_labels(esgify):
    model, tokenizer = esgify
    [labels] = classify(["The factory released toxic waste into the river, killing thousands of fish."],
                        model, tokenizer)
    top_labels = {label for label, _ in labels}
    assert top_labels & {"Biodiversity", "Wastewater Management", "Surface Water Pollution"}


@pytest.mark.slow
def test_scores_are_probabilities(esgify):
    model, tokenizer = esgify
    [labels] = classify(["Some text."], model, tokenizer, top_k=47)
    assert all(0.0 <= score <= 1.0 for _, score in labels)


@pytest.mark.slow
def test_result_does_not_depend_on_batch(esgify):
    """A text's scores must not change based on which other texts share its batch.

    This would fail if the model were left in training mode (BatchNorm would use
    batch statistics) or if padding leaked into the mean pooling.
    """
    model, tokenizer = esgify
    text = "She faced employment rejection because of her gender."
    alone = classify([text], model, tokenizer)[0]
    batched = classify([text, "A much longer text about something else entirely, to force padding."],
                       model, tokenizer)[0]
    assert alone == batched
