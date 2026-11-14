"""End-to-end ESG risk scoring of a project description."""

import numpy as np

from esg_screening.esgify.chunking import make_windows, split_sentences
from esg_screening.esgify.classifier import load_model, predict_scores
from esg_screening.esgify.masking import NameMasker
from esg_screening.esgify.scoring import ScoringConfig, score_document


def pick_evidence(chunk_scores, sentence_scores, windows, sentences):
    """For each label, the sentence that best explains its score.

    The label's highest-scoring window decides where to look; within that window, the
    sentence with the highest score of its own for the label is the evidence.
    """
    evidence = []
    for label_idx, chunk_idx in enumerate(chunk_scores.argmax(axis=0)):
        start, end = windows[chunk_idx]
        best = start + int(sentence_scores[start:end, label_idx].argmax())
        evidence.append(sentences[best])
    return evidence


class ESGRiskScorer:
    """Scores the ESG risk of a free-text project description.

    Pipeline: split into sentences -> mask names in each sentence -> group sentences into
    overlapping windows -> score each window with ESGify -> aggregate into one result.
    Single sentences are also scored, only to pick the evidence shown for each risk.

    Loading the models takes a few seconds, so create one scorer and reuse it.
    """

    def __init__(self, model, tokenizer, masker=None, config=ScoringConfig()):
        self.model = model
        self.tokenizer = tokenizer
        self.masker = masker
        self.config = config
        self.labels = [model.id2label[i] for i in range(len(model.id2label))]

    @classmethod
    def load(cls, config=ScoringConfig(), mask_entities=True):
        """Create a scorer with the pretrained ESGify model and, optionally, the name masker."""
        model, tokenizer = load_model()
        masker = NameMasker() if mask_entities else None
        return cls(model, tokenizer, masker, config)

    def score(self, text, with_evidence=True):
        return self.score_many([text], with_evidence)[0]

    def score_many(self, texts, with_evidence=True):
        """Score several descriptions at once.

        Faster than calling score() for each text, because both models process the inputs of
        all texts together. With with_evidence=False the single-sentence scoring used only to
        pick evidence is skipped (evidence stays empty).
        """
        split = [split_sentences(text) for text in texts]
        everything = [sentence for sentences in split for sentence in sentences]
        masked = self.masker.mask(everything) if self.masker and everything else everything

        # Collect the model inputs of every text (its windows, plus its sentences if evidence is
        # wanted) into one list, remembering where each text's rows start.
        rows, plans, start = [], [], 0
        for sentences in split:
            model_sentences = masked[start:start + len(sentences)]
            start += len(sentences)
            windows = make_windows(len(sentences), self.config.window_size)
            plans.append((sentences, windows, len(rows)))
            rows += [" ".join(model_sentences[a:b]) for a, b in windows]
            rows += model_sentences if with_evidence else []

        scores = (predict_scores(rows, self.model, self.tokenizer, self.config.batch_size).numpy()
                  if rows else np.zeros((0, len(self.labels))))

        results = []
        for sentences, windows, first_row in plans:
            chunk_scores = scores[first_row:first_row + len(windows)]
            sentence_scores = scores[first_row + len(windows):first_row + len(windows) + len(sentences)]
            evidence = (pick_evidence(chunk_scores, sentence_scores, windows, sentences)
                        if with_evidence and sentences else [""] * len(self.labels))
            results.append(score_document(chunk_scores, self.labels, evidence, self.config,
                                          n_sentences=len(sentences)))
        return results
