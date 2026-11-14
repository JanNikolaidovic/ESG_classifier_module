"""Shared test fixtures. The real models are loaded once per test session and reused."""

import pytest

from esg_screening.esgify.classifier import load_model
from esg_screening.esgify.masking import NameMasker
from esg_screening.esgify.pipeline import ESGRiskScorer


@pytest.fixture(scope="session")
def esgify():
    return load_model()


@pytest.fixture(scope="session")
def masker():
    return NameMasker()


@pytest.fixture(scope="session")
def scorer(esgify, masker):
    model, tokenizer = esgify
    return ESGRiskScorer(model, tokenizer, masker)
