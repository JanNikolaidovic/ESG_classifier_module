"""Step 3: run ESGify on every IFC project description (the inputs of approach (a)).

For each description we run the same scorer as `score-esg` and keep:
- the 47 ESGify class scores (highest score across chunks), one column per class
- the hand-made risk score and E/S/G sub-scores, to compare against
Results are added to the CSV every 25 projects, so the run can be stopped and resumed.
Each row stores a fingerprint of the description, so a changed description is scored again.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

from esg_screening.ifc.dataset import DATASET_PATH

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/ifc_esgify_features.csv")
CLASS_PREFIX = "class: "  # column name prefix of the 47 ESGify class scores
SAVE_EVERY = 25


def fingerprint(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def result_to_row(result):
    """Flatten an ESGRiskResult into one table row."""
    return {
        **result.to_features(),
        "n_sentences": result.n_sentences,
        **{CLASS_PREFIX + label: score for label, score in result.class_scores.items()},
    }


def compute_features(dataset_path=DATASET_PATH, features_path=FEATURES_PATH, limit=None, scorer=None):
    projects = pd.read_csv(dataset_path, dtype={"project_number": str})
    projects["fingerprint"] = projects["description"].map(fingerprint)

    done = set()
    if features_path.exists():
        done = set(pd.read_csv(features_path, usecols=["fingerprint"])["fingerprint"])
    # Random order (fixed seed): if the run is stopped early, the projects scored so far are
    # still a fair sample of the whole dataset, so an early evaluation is meaningful.
    missing = projects[~projects["fingerprint"].isin(done)].sample(frac=1, random_state=0)
    todo = missing.head(limit)
    log.info("%d projects already scored, scoring %d", len(projects) - len(missing), len(todo))
    if todo.empty:
        return

    if scorer is None:
        from esg_screening.esgify import ESGRiskScorer  # slow import, only when needed
        scorer = ESGRiskScorer.load()

    features_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), SAVE_EVERY):
        batch = todo.iloc[start:start + SAVE_EVERY]
        results = scorer.score_many(batch["description"].tolist(), with_evidence=False)
        rows = [
            {"project_number": p.project_number, "fingerprint": p.fingerprint, **result_to_row(result)}
            for p, result in zip(batch.itertuples(), results)
        ]
        pd.DataFrame(rows).to_csv(features_path, mode="a", header=not features_path.exists(), index=False)
        log.info("%d / %d projects scored", start + len(batch), len(todo))
