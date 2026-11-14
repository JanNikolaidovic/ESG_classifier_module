from pathlib import Path

import numpy as np
import pytest

from esg_screening.esgify import taxonomy
from esg_screening.esgify.pipeline import pick_evidence

SAMPLES = Path(__file__).parents[2] / "data" / "samples"


def test_pick_evidence_uses_best_sentence_of_best_window():
    sentences = ["a", "b", "c", "d"]
    windows = [(0, 2), (1, 3), (2, 4)]
    chunk_scores = np.array([[0.1], [0.9], [0.2]])           # window (1, 3) wins
    sentence_scores = np.array([[0.99], [0.3], [0.8], [0.1]])  # "a" is higher, but outside it
    assert pick_evidence(chunk_scores, sentence_scores, windows, sentences) == ["c"]


# ---------- integration tests (slow, real models) ----------

@pytest.fixture(scope="module")
def sample_results(scorer):
    names = ["rooftop_solar_project", "dairy_plant_expansion", "hydropower_project"]
    return {name: scorer.score((SAMPLES / f"{name}.txt").read_text()) for name in names}


@pytest.mark.slow
def test_samples_ranked_low_medium_high(sample_results):
    low = sample_results["rooftop_solar_project"].esg_risk_score
    medium = sample_results["dairy_plant_expansion"].esg_risk_score
    high = sample_results["hydropower_project"].esg_risk_score
    assert low < 15 < medium < 70 < high


@pytest.mark.slow
def test_low_risk_project_has_no_severe_flags(sample_results):
    flags = sample_results["rooftop_solar_project"].flagged_risks
    assert all(risk.weight < taxonomy.HIGH for risk in flags)


@pytest.mark.slow
def test_key_hydropower_risks_found_with_evidence(sample_results):
    flags = {risk.label: risk for risk in sample_results["hydropower_project"].flagged_risks}
    assert "indigenous group" in flags["Indigenous People"].evidence
    assert "anti-corruption" in flags["Economic Crime"].evidence
    assert "relocated" in flags["Land Acquisition and Resettlement (E)"].evidence


@pytest.mark.slow
def test_scoring_is_deterministic(scorer):
    text = (SAMPLES / "dairy_plant_expansion.txt").read_text()
    assert scorer.score(text).to_dict() == scorer.score(text).to_dict()


@pytest.mark.slow
def test_score_many_matches_score(scorer):
    texts = [(SAMPLES / f"{name}.txt").read_text() for name in ["dairy_plant_expansion", "rooftop_solar_project"]]
    together = scorer.score_many(texts + [""])
    one_by_one = [scorer.score(text) for text in texts + [""]]
    for a, b in zip(together, one_by_one):
        assert a.class_scores == pytest.approx(b.class_scores, abs=1e-5)
        assert a.esg_risk_score == pytest.approx(b.esg_risk_score, abs=1e-3)


@pytest.mark.slow
def test_empty_text(scorer):
    result = scorer.score("")
    assert result.esg_risk_score == 0.0
    assert result.n_sentences == 0
