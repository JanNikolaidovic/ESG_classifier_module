"""Step 4: how well can we predict IFC's category (A/B/C) from a project description?

Every approach is tested the same way, with 5-fold cross-validation: the projects are split
into 5 groups; each group in turn is hidden, the approach learns from the other 4 groups and
then predicts the hidden one. So every project is predicted exactly once, by a model that never
saw it. Projects of the same company always stay in the same group, so an approach cannot score
well just by recognising a company it has already seen.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from esg_screening.categorizer.features import CLASS_PREFIX, FEATURES_PATH, fingerprint, result_to_row
from esg_screening.ifc.dataset import DATASET_PATH

log = logging.getLogger(__name__)

REPORT_PATH = Path("reports/ifc_category_evaluation.md")
SAMPLES_DIR = Path("data/samples")
CATEGORIES = ["A", "B", "C"]
HAND_MADE_COLUMNS = ["esg_risk_score", "esg_e_score", "esg_s_score", "esg_g_score"]
APPROACH_A = "(a) ESGify 47 class scores"
WORD_COUNTS = "Word counts (TF-IDF), no ESGify"
DETAILED = [APPROACH_A, WORD_COUNTS]  # approaches explained in detail in the report

# Our owner-style samples: the IFC industry they would belong to, and the category we expect
SAMPLE_INDUSTRIES = {
    "dairy_plant_expansion.txt": "Agribusiness and Forestry",
    "hydropower_project.txt": "Infrastructure",
    "rooftop_solar_project.txt": "Infrastructure",
    "solar_with_risk_denials.txt": "Infrastructure",
}
SAMPLE_EXPECTED = {
    "dairy_plant_expansion.txt": "B",
    "hydropower_project.txt": "A",
    "rooftop_solar_project.txt": "B or C",
    "solar_with_risk_denials.txt": "B or C",
}


def logistic_regression():
    # class_weight="balanced": A is rare, so mistakes on A projects count more during learning
    return LogisticRegression(class_weight="balanced", max_iter=5000)


def build_approaches(class_columns):
    """Each approach = which input columns it uses + a model. All are scikit-learn pipelines."""
    def using(*transformers):
        return ColumnTransformer(list(transformers))

    industry = ("industry", OneHotEncoder(handle_unknown="ignore"), ["industry"])
    esgify = ("esgify", StandardScaler(), class_columns)
    return {
        "Always B (most common category)": make_pipeline(
            using(("any", "passthrough", ["esg_risk_score"])), DummyClassifier(strategy="most_frequent")),
        "Industry only (no text)": make_pipeline(using(industry), logistic_regression()),
        "Hand-made ESG risk score": make_pipeline(
            using(("score", StandardScaler(), HAND_MADE_COLUMNS)), logistic_regression()),
        APPROACH_A: make_pipeline(using(esgify), logistic_regression()),
        "(a) + industry": make_pipeline(using(esgify, industry), logistic_regression()),
        WORD_COUNTS: make_pipeline(
            using(("words", TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True,
                                            stop_words="english"), "description")),
            logistic_regression()),
    }


def summarise(y_true, probabilities):
    """Plain-language metrics, computed from each project's predicted A/B/C probabilities."""
    predicted = np.array(CATEGORIES)[probabilities.argmax(axis=1)]
    is_a, chance_of_a = y_true == "A", probabilities[:, CATEGORIES.index("A")]
    precision, recall, _ = precision_recall_curve(is_a, chance_of_a)
    return {
        "Macro F1": f1_score(y_true, predicted, average="macro"),
        "Balanced accuracy": balanced_accuracy_score(y_true, predicted),
        "A found (recall)": recall_score(y_true, predicted, labels=["A"], average=None)[0],
        "A correct (precision)": precision_score(y_true, predicted, labels=["A"], average=None, zero_division=0)[0],
        "A ranking (ROC AUC)": roc_auc_score(is_a, chance_of_a),
        # Lower the bar for "A" until 80% of real A projects are flagged: what share of flags are right?
        "A correct when 80% of A found": precision[recall >= 0.8].max(),
    }


def load_data(dataset_path=DATASET_PATH, features_path=FEATURES_PATH):
    """IFC projects joined with their ESGify features (only projects that have both)."""
    projects = pd.read_csv(dataset_path, dtype={"project_number": str})
    projects["fingerprint"] = projects["description"].map(fingerprint)
    features = pd.read_csv(features_path, dtype={"project_number": str}).drop_duplicates("fingerprint")
    return projects.merge(features.drop(columns="project_number"), on="fingerprint")


def run_evaluation(report_path=REPORT_PATH):
    data = load_data()
    class_columns = [c for c in data.columns if c.startswith(CLASS_PREFIX)]
    y, companies = data["category"].to_numpy(), data["company_name"].fillna(data["project_number"])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    log.info("Evaluating on %d projects: %s", len(data), data["category"].value_counts().to_dict())

    results, all_probabilities = {}, {}
    for name, pipeline in build_approaches(class_columns).items():
        probabilities = cross_val_predict(pipeline, data, y, cv=folds, groups=companies, method="predict_proba")
        results[name] = summarise(y, probabilities)
        all_probabilities[name] = probabilities
        log.info("%-35s %s", name, {k: round(v, 3) for k, v in results[name].items()})

    # Train every approach on all projects, to see what they learned and to try our own samples
    models = {name: pipeline.fit(data, y) for name, pipeline in build_approaches(class_columns).items()}
    report = write_report(data, y, results, all_probabilities, models)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    log.info("Report saved to %s", report_path)


def strongest_signals(model, n=10):
    """Inputs (ESGify topics or words) that push predictions most towards / away from Category A."""
    names = [name.split("__", 1)[1].removeprefix(CLASS_PREFIX) for name in model[0].get_feature_names_out()]
    weights = model[-1].coef_[list(model[-1].classes_).index("A")]
    order = np.argsort(weights)
    return [(names[i], weights[i]) for i in order[::-1][:n]], [(names[i], weights[i]) for i in order[:n]]


def sample_inputs():
    """The owner-style example descriptions in data/samples/, with everything the approaches need."""
    from esg_screening.esgify import ESGRiskScorer
    paths = sorted(SAMPLES_DIR.glob("*.txt"))
    texts = [path.read_text(encoding="utf-8") for path in paths]
    results = ESGRiskScorer.load().score_many(texts, with_evidence=False)
    rows = [
        {"description": text, "industry": SAMPLE_INDUSTRIES.get(path.name, "unknown"), **result_to_row(result)}
        for path, text, result in zip(paths, texts, results)
    ]
    return [path.name for path in paths], pd.DataFrame(rows)


def write_report(data, y, results, all_probabilities, models):
    counts = data["category"].value_counts()
    lines = [
        "# Predicting IFC's E&S category from the project description",
        "",
        f"Data: {len(data)} IFC projects with a description "
        f"(A: {counts.get('A', 0)}, B: {counts.get('B', 0)}, C: {counts.get('C', 0)}). "
        "Test method: 5-fold cross-validation, projects of the same company kept together.",
        "",
        "## Results",
        "",
        "| Approach | " + " | ".join(next(iter(results.values()))) + " |",
        "|---|" + "---|" * len(next(iter(results.values()))),
    ]
    for name, metrics in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in metrics.values()) + " |")

    lines += [
        "",
        "What the columns mean (all between 0 and 1, higher is better):",
        "- **Macro F1**: overall quality, with A, B and C counting equally however rare they are.",
        "- **Balanced accuracy**: the average of the share of A, B and C projects predicted correctly. Guessing gives 0.33.",
        "- **A found (recall)**: of the real Category A projects, the share predicted as A. Missing an A is the costly mistake.",
        "- **A correct (precision)**: of the projects predicted as A, the share that really are A.",
        "- **A ranking (ROC AUC)**: if we sort projects by predicted chance of being A, how often a real A comes before a non-A. 0.5 = random, 1 = perfect.",
        "- **A correct when 80% of A found**: we lower the bar for flagging \"A\" until 80% of the real A projects are flagged; this is the share of flagged projects that really are A. It compares approaches fairly for screening, where catching most A projects matters most.",
    ]

    for name in DETAILED:
        lines += ["", f"## {name}: in detail", "",
                  "Confusion matrix (rows: real category, columns: predicted):", "",
                  "| real \\ predicted | A | B | C |", "|---|---|---|---|"]
        predicted = np.array(CATEGORIES)[all_probabilities[name].argmax(axis=1)]
        for category, row in zip(CATEGORIES, confusion_matrix(y, predicted, labels=CATEGORIES)):
            lines.append(f"| {category} | " + " | ".join(str(v) for v in row) + " |")
        towards, away = strongest_signals(models[name])
        lines += ["", "Inputs that push most **towards** Category A: "
                  + ", ".join(f"{label} ({weight:+.2f})" for label, weight in towards)]
        lines += ["", "Inputs that push most **away** from Category A: "
                  + ", ".join(f"{label} ({weight:+.2f})" for label, weight in away)]

    names, samples = sample_inputs()
    shown = [name for name in models if name != "Always B (most common category)"]
    lines += ["", "## Our own example descriptions (owner-style text, not IFC)", "",
              "Each cell: predicted category (probability of A). Industries were set by hand for these samples.", "",
              "| Sample | Expected (our judgement) | " + " | ".join(shown) + " |",
              "|---|---|" + "---|" * len(shown)]
    for i, name in enumerate(names):
        cells = []
        for approach in shown:
            probabilities = models[approach].predict_proba(samples.iloc[[i]])[0]
            cells.append(f"{CATEGORIES[probabilities.argmax()]} ({probabilities[0]:.2f})")
        lines.append(f"| {name} | {SAMPLE_EXPECTED.get(name, '?')} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
