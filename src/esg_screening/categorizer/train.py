"""Train the category model on all IFC projects and save it (command: `train-category`).

Before training on everything, the model is tested once with the same 5-fold cross-validation
as the evaluation. The test results are saved with the model, so every saved model carries
its own report card.
"""

import hashlib
import logging

import pandas as pd

from esg_screening.categorizer import model as category_model
from esg_screening.categorizer.evaluate import cross_validated_probabilities, summarise
from esg_screening.ifc.dataset import DATASET_PATH

log = logging.getLogger(__name__)


def train_and_save(dataset_path=DATASET_PATH, model_path=category_model.MODEL_PATH):
    data = pd.read_csv(dataset_path, dtype={"project_number": str})

    log.info("Testing the model with 5-fold cross-validation on %d projects", len(data))
    probabilities = cross_validated_probabilities(category_model.word_count_model(), data)
    metrics = summarise(data["category"].to_numpy(), probabilities)

    log.info("Training on all %d projects", len(data))
    model = category_model.word_count_model().fit(data, data["category"])
    category_model.save(model, {
        "training_data": {
            "file": str(dataset_path),
            "fingerprint": hashlib.sha1(dataset_path.read_bytes()).hexdigest()[:16],
            "projects": len(data),
            "categories": {c: int(n) for c, n in data["category"].value_counts().items()},
        },
        "cross_validation": {name: round(float(value), 3) for name, value in metrics.items()},
    }, model_path)
    log.info("Model saved to %s", model_path)
    return model, metrics
