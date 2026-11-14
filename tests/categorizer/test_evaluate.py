"""Tests for the evaluation code, on a tiny made-up dataset where the answer is obvious."""

import numpy as np
import pandas as pd
import pytest

from esg_screening.categorizer.evaluate import (APPROACH_A, HAND_MADE_COLUMNS, WORD_COUNTS, build_approaches,
                                                strongest_signals, summarise)
from esg_screening.categorizer.features import CLASS_PREFIX

CLASS_COLUMNS = [CLASS_PREFIX + "Biodiversity", CLASS_PREFIX + "Disclosure"]


def test_summarise_perfect_predictions():
    y = np.array(["A", "B", "C"])
    metrics = summarise(y, np.eye(3))  # each project gets probability 1 for its real category
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def toy_data():
    """60 projects: Category A projects have a high Biodiversity score, C projects a low one."""
    rng = np.random.default_rng(0)
    category = np.repeat(["A", "B", "C"], 20)
    biodiversity = np.select([category == "A", category == "B"], [0.9, 0.5], 0.1) + rng.normal(0, 0.05, 60)
    data = pd.DataFrame({
        CLASS_COLUMNS[0]: biodiversity,
        CLASS_COLUMNS[1]: rng.uniform(size=60),
        "industry": rng.choice(["Mining", "Banking"], 60),
        "description": np.where(category == "A", "forest habitat species", "office software services"),
        **{column: rng.uniform(size=60) for column in HAND_MADE_COLUMNS},
    })
    return data, category


def test_every_approach_trains_and_predicts():
    data, y = toy_data()
    for name, pipeline in build_approaches(CLASS_COLUMNS).items():
        probabilities = pipeline.fit(data, y).predict_proba(data)
        assert probabilities.shape == (60, 3), name


def test_approach_a_learns_an_obvious_signal():
    data, y = toy_data()
    pipeline = build_approaches(CLASS_COLUMNS)[APPROACH_A]
    metrics = summarise(y, pipeline.fit(data, y).predict_proba(data))
    assert metrics["Balanced accuracy"] > 0.9


def test_strongest_signals_names_the_real_signal():
    data, y = toy_data()
    approaches = build_approaches(CLASS_COLUMNS)
    towards, _ = strongest_signals(approaches[APPROACH_A].fit(data, y), n=1)
    assert towards[0][0] == "Biodiversity"
    towards, _ = strongest_signals(approaches[WORD_COUNTS].fit(data, y), n=3)
    assert {"forest", "habitat", "species"} & {word for word, _ in towards}
