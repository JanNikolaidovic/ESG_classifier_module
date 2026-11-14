"""Tests for training, saving, loading and using the category model, on a tiny made-up dataset."""

import pandas as pd
import pytest

from esg_screening.categorizer import model as category_model
from esg_screening.categorizer.train import train_and_save

WORDS = {
    "A": "construction of a new gold mine and dam on the river with a km long road",
    "B": "expansion of a food factory producing packaged products for local shops",
    "C": "a bank offering loans and software services to small companies",
}


def toy_table(per_category=20):
    rows = [
        {"category": c, "description": f"{text} number {i}", "company_name": f"{c}-company-{i}",
         "project_number": f"{c}{i}"}
        for c, text in WORDS.items() for i in range(per_category)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def trained():
    table = toy_table()
    return category_model.word_count_model().fit(table, table["category"])


def test_predicts_the_obvious_category(trained):
    [(category, probabilities)] = category_model.predict(trained, ["a new gold mine with a dam on the river"])
    assert category == "A"
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_explain_names_words_from_the_text(trained):
    towards, away = category_model.explain(trained, "a gold mine financed by a bank loan")
    assert "mine" in [word for word, _ in towards]
    assert "bank" in [word for word, _ in away]


def test_save_and_load_give_the_same_predictions(trained, tmp_path):
    path = tmp_path / "model.joblib"
    category_model.save(trained, {"note": "test"}, path)
    loaded, metadata = category_model.load(path)
    text = ["software services for a bank"]
    assert category_model.predict(loaded, text) == category_model.predict(trained, text)
    assert metadata["note"] == "test"
    assert {"trained_at", "versions", "model"} <= set(metadata)


def test_missing_model_gives_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-category"):
        category_model.load(tmp_path / "nothing.joblib")


def test_train_and_save_stores_test_results(tmp_path):
    dataset = tmp_path / "projects.csv"
    toy_table().to_csv(dataset, index=False)
    train_and_save(dataset_path=dataset, model_path=tmp_path / "model.joblib")
    _, metadata = category_model.load(tmp_path / "model.joblib")
    assert metadata["training_data"]["projects"] == 60
    assert metadata["cross_validation"]["Balanced accuracy"] > 0.9
