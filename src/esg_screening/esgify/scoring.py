"""Turn chunk-level ESGify scores into a document-level ESG risk score.

Steps:
1. Document score per class = highest score across chunks. A risk counts if any part
   of the description raises it, however long the rest of the text is.
2. Flag strength per class = how far that score is above `flag_threshold`, scaled to 0-1.
   ESGify gives 0.5-0.8 to many unrelated classes, so only high scores count.
3. Risk load = sum of severity weight x flag strength. Roughly "the number of serious
   risks the text clearly raises".
4. Risk score (0-100) = 100 x (1 - exp(-load / saturation)). Grows with every extra risk
   but never exceeds 100; a load equal to `saturation` gives 63.

Pillar (E/S/G) scores use the same steps restricted to that pillar's classes.
"""

from dataclasses import asdict, dataclass

import numpy as np

from esg_screening.esgify import taxonomy

# Bump whenever anything that changes the scores changes (ESGify model, masking, chunking,
# weights or formula). Models trained on ESG features must be retrained when it changes.
SCORER_VERSION = "0.1.0"


@dataclass(frozen=True)
class ScoringConfig:
    window_size: int = 3          # sentences per chunk
    flag_threshold: float = 0.8   # class scores below this count as zero risk
    saturation: float = 5.0       # risk load that maps to a score of 63
    batch_size: int = 16          # chunks per model call


@dataclass
class FlaggedRisk:
    label: str
    pillar: str
    score: float         # highest model score for this class across chunks
    weight: float        # severity weight from the taxonomy
    contribution: float  # weight x flag strength, this risk's share of the risk load
    evidence: str        # original (unmasked) sentence that most supports this risk


@dataclass
class ESGRiskResult:
    esg_risk_score: float   # 0-100
    pillar_scores: dict     # {"E": 0-100, "S": 0-100, "G": 0-100}
    risk_load: float
    flagged_risks: list     # FlaggedRisk, largest contribution first
    class_scores: dict      # every label -> highest score across chunks
    n_sentences: int
    n_chunks: int
    scorer_version: str = SCORER_VERSION

    def to_features(self):
        """Flat numeric summary, e.g. for a downstream model."""
        return {
            "esg_risk_score": self.esg_risk_score,
            "esg_e_score": self.pillar_scores[taxonomy.ENVIRONMENTAL],
            "esg_s_score": self.pillar_scores[taxonomy.SOCIAL],
            "esg_g_score": self.pillar_scores[taxonomy.GOVERNANCE],
            "esg_n_flags": len(self.flagged_risks),
        }

    def to_dict(self):
        return asdict(self)


def flag_strength(score, threshold):
    """0 at or below the threshold, rising linearly to 1 at a score of 1."""
    return float(np.clip((score - threshold) / (1.0 - threshold), 0.0, 1.0))


def load_to_score(load, saturation):
    return float(100.0 * (1.0 - np.exp(-load / saturation)))


def score_document(chunk_scores, labels, evidence, config, n_sentences):
    """Build an ESGRiskResult from a (n_chunks, n_labels) array of model scores.

    labels[i] names column i; evidence[i] is the supporting text for that label.
    """
    n_chunks = len(chunk_scores)
    doc_scores = chunk_scores.max(axis=0) if n_chunks else np.zeros(len(labels))

    flagged = []
    for i, label in enumerate(labels):
        weight = taxonomy.weight_of(label)
        contribution = weight * flag_strength(doc_scores[i], config.flag_threshold)
        if contribution > 0:
            flagged.append(FlaggedRisk(
                label=label,
                pillar=taxonomy.pillar_of(label),
                score=float(doc_scores[i]),
                weight=weight,
                contribution=contribution,
                evidence=evidence[i],
            ))
    flagged.sort(key=lambda risk: risk.contribution, reverse=True)

    risk_load = sum(risk.contribution for risk in flagged)
    pillar_loads = {
        pillar: sum(risk.contribution for risk in flagged if risk.pillar == pillar)
        for pillar in taxonomy.PILLARS
    }
    return ESGRiskResult(
        esg_risk_score=load_to_score(risk_load, config.saturation),
        pillar_scores={p: load_to_score(load, config.saturation) for p, load in pillar_loads.items()},
        risk_load=risk_load,
        flagged_risks=flagged,
        class_scores={label: float(s) for label, s in zip(labels, doc_scores)},
        n_sentences=n_sentences,
        n_chunks=n_chunks,
    )
