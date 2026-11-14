"""The category model: predicts a project's E&S category (A/B/C) from its description.

It is the best approach from the IFC evaluation: word counts (TF-IDF) + logistic regression.
- TF-IDF turns a text into numbers: one per word or word pair, higher for words that appear in
  this text but are rare in other texts.
- Logistic regression learned one weight per word from 3,994 IFC projects and turns the weighted
  sum into a probability for A, B and C.

The trained model is saved in models/ as two files: the model itself (.joblib) and a small
description of it (.json: when and on what data it was trained, and how well it tested).
"""

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

log = logging.getLogger(__name__)

CATEGORIES = ["A", "B", "C"]
MEANINGS = {
    "A": "potentially significant, diverse or irreversible environmental and social impacts",
    "B": "limited impacts: few, site-specific, mostly reversible, manageable with standard measures",
    "C": "minimal or no environmental and social impacts",
}
MODEL_PATH = Path("models/category_model.joblib")
NAME = "Word counts (TF-IDF) + logistic regression"


def logistic_regression():
    # class_weight="balanced": A is rare, so mistakes on A projects count more during learning
    return LogisticRegression(class_weight="balanced", max_iter=5000)


# Very common words ("the", "and", ...) carry no meaning and are ignored. scikit-learn's standard
# list also contains "mine" (as in "that's mine"), "well" and "fire", which matter here: a gold
# mine, an oil well, fire safety. So those three are kept.
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - {"mine", "well", "fire"})


def word_count_model():
    """A new, untrained category model. Its input is a table with a "description" column."""
    words = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP_WORDS)
    return make_pipeline(ColumnTransformer([("words", words, "description")]), logistic_regression())


def as_input(texts):
    """The table the model expects: one row per description."""
    return pd.DataFrame({"description": list(texts)})


def save(model, metadata, path=MODEL_PATH):
    """Save the model and its description (same name, .json) next to each other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    metadata = {
        "model": NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metadata,
        "versions": {"scikit-learn": sklearn.__version__, "python": platform.python_version()},
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load(path=MODEL_PATH):
    """Load a saved model and its description.

    Only load model files you created yourself, or the published one after checking its SHA-256
    checksum against the README: a .joblib file can run any code when loaded.
    """
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python main.py train-category` first.")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["versions"]["scikit-learn"] != sklearn.__version__:
        log.warning("The model was saved with scikit-learn %s but %s is installed; "
                    "retrain it if predictions look wrong.", metadata["versions"]["scikit-learn"], sklearn.__version__)
    return joblib.load(path), metadata


def predict(model, texts):
    """For each text: (most likely category, {category: probability})."""
    probabilities = model.predict_proba(as_input(texts))
    classes = list(model[-1].classes_)
    return [
        (classes[int(row.argmax())], {c: float(p) for c, p in zip(classes, row)})
        for row in probabilities
    ]


def explain(model, text, n=8):
    """The words of this text that push its prediction most towards and most away from Category A.

    Each word's push = how strongly it appears in the text (TF-IDF) x the weight the model learned for it.
    """
    row = model[0].transform(as_input([text])).tocsr()
    names = model[0].get_feature_names_out()
    weights = model[-1].coef_[list(model[-1].classes_).index("A")]
    pushes = {names[i].split("__", 1)[1]: value * weights[i] for i, value in zip(row.indices, row.data)}
    ranked = sorted(pushes.items(), key=lambda item: item[1])
    towards = [(word, push) for word, push in reversed(ranked) if push > 0][:n]
    away = [(word, push) for word, push in ranked if push < 0][:n]
    return towards, away
