"""Tests for the scoring math, using hand-made score arrays instead of the model."""

import math

import numpy as np
import pytest

from esg_screening.esgify import taxonomy
from esg_screening.esgify.scoring import ScoringConfig, flag_strength, load_to_score, score_document

LABELS = list(taxonomy.LABELS)
CONFIG = ScoringConfig()


def make_scores(n_chunks, values):
    """(n_chunks, n_labels) array of 0.1s, with {(chunk, label): score} overrides."""
    scores = np.full((n_chunks, len(LABELS)), 0.1)
    for (chunk, label), value in values.items():
        scores[chunk, LABELS.index(label)] = value
    return scores


def score(chunk_scores):
    evidence = [f"evidence for {label}" for label in LABELS]
    return score_document(chunk_scores, LABELS, evidence, CONFIG, n_sentences=len(chunk_scores))


# ---------- building blocks ----------

@pytest.mark.parametrize("value, expected", [(0.5, 0.0), (0.8, 0.0), (0.9, 0.5), (1.0, 1.0)])
def test_flag_strength(value, expected):
    assert flag_strength(value, 0.8) == pytest.approx(expected)


def test_load_to_score():
    assert load_to_score(0.0, 5.0) == 0.0
    assert load_to_score(5.0, 5.0) == pytest.approx(100 * (1 - math.exp(-1)))  # ~63.2
    assert load_to_score(1000.0, 5.0) <= 100.0


# ---------- document scoring ----------

def test_text_without_risks_scores_zero():
    result = score(make_scores(4, {}))
    assert result.esg_risk_score == 0.0
    assert result.flagged_risks == []


def test_empty_document_scores_zero():
    result = score_document(np.zeros((0, len(LABELS))), LABELS, [], CONFIG, n_sentences=0)
    assert result.esg_risk_score == 0.0
    assert result.n_chunks == 0


def test_highest_chunk_score_is_used():
    result = score(make_scores(3, {(0, "Biodiversity"): 0.3, (2, "Biodiversity"): 0.95}))
    assert result.class_scores["Biodiversity"] == pytest.approx(0.95)
    assert [r.label for r in result.flagged_risks] == ["Biodiversity"]


def test_benefit_class_is_not_a_risk():
    result = score(make_scores(1, {(0, "Energy Efficiency and Renewables"): 1.0}))
    assert result.esg_risk_score == 0.0


def test_scores_go_to_the_right_pillar():
    result = score(make_scores(1, {(0, "Economic Crime"): 0.95}))
    assert result.pillar_scores[taxonomy.GOVERNANCE] > 0
    assert result.pillar_scores[taxonomy.ENVIRONMENTAL] == 0
    assert result.pillar_scores[taxonomy.SOCIAL] == 0


def test_more_risks_give_a_higher_score():
    one = score(make_scores(1, {(0, "Biodiversity"): 0.95}))
    two = score(make_scores(1, {(0, "Biodiversity"): 0.95, (0, "Forced Labour"): 0.95}))
    assert two.esg_risk_score > one.esg_risk_score


def test_severe_risk_counts_more_than_management_topic():
    severe = score(make_scores(1, {(0, "Forced Labour"): 0.95}))
    management = score(make_scores(1, {(0, "Disclosure"): 0.95}))
    assert severe.esg_risk_score > management.esg_risk_score


def test_flagged_risks_sorted_and_carry_evidence():
    result = score(make_scores(1, {(0, "Disclosure"): 0.99, (0, "Forced Labour"): 0.9}))
    contributions = [r.contribution for r in result.flagged_risks]
    assert contributions == sorted(contributions, reverse=True)
    assert result.flagged_risks[0].evidence == "evidence for Forced Labour"


def test_to_features():
    features = score(make_scores(1, {(0, "Biodiversity"): 0.95})).to_features()
    assert set(features) == {"esg_risk_score", "esg_e_score", "esg_s_score", "esg_g_score", "esg_n_flags"}
    assert features["esg_n_flags"] == 1
