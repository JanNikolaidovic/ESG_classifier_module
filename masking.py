"""Mask named entities before ESG classification.

ESGify was trained on texts where names were replaced by tags, so that the model
learns risks from what happened rather than from who it happened to. This is based
on the preprocessing notebook linked from the ESGify README: run the FLAIR OntoNotes
NER tagger, replace confident entities with <TAG>, and remove English stopwords.

The notebook removes stopwords *before* NER; we do it after, because the tagger
relies on words like "in" and "the" and loses confidence without them.
"""

import re

import nltk
from flair.data import Sentence
from flair.nn import Classifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

NER_MODEL_NAME = "flair/ner-english-ontonotes"
# The notebook masks only FAC, LOC, ORG and PERSON; GPE (countries, cities, states) and DATE
# are added because the ESGify README says countries and dates were masked in training.
DEFAULT_TAGS = (
    "FAC",     # facilities: buildings, airports, bridges
    "LOC",     # non-political locations: rivers, mountains, regions
    "ORG",     # organisations: companies, agencies
    "PERSON",  # people
    "GPE",     # countries, cities, states
    "DATE",    # dates and periods: "Monday", "2023", "last year"
)
TAG_PATTERN = re.compile(r"(<[A-Z_]+>)")


def _ensure_nltk_data():
    """Download the NLTK stopword list and tokenizer data on first use."""
    for resource, path in [("stopwords", "corpora/stopwords"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


class NameMasker:
    """Replaces named entities in text with tags such as <ORG> or <PERSON>."""

    # The notebook used 0.8, but real entities often score 0.75-0.85 (e.g. "K&W Company" 0.77),
    # while in our checks no non-entity scored above 0.7.
    def __init__(self, tags=DEFAULT_TAGS, min_confidence=0.7, remove_stopwords=True,
                 model_name=NER_MODEL_NAME):
        _ensure_nltk_data()
        self.tagger = Classifier.load(model_name)
        self.tags = set(tags)
        self.min_confidence = min_confidence
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()

    def mask(self, texts):
        """Return a masked copy of each text in the list."""
        sentences = [Sentence(text) for text in texts]
        self.tagger.predict(sentences)  # one batched call is faster than one call per text
        masked = [self._replace_entities(text, sentence) for text, sentence in zip(texts, sentences)]
        return [self._remove_stopwords(text) for text in masked]

    def _remove_stopwords(self, text):
        """Drop stopwords, keeping <TAG> tokens intact (word_tokenize would split them)."""
        words = []
        for piece in TAG_PATTERN.split(text):
            if TAG_PATTERN.fullmatch(piece):
                words.append(piece)
            else:
                words.extend(w for w in word_tokenize(piece) if w.lower() not in self.stop_words)
        return " ".join(words)

    def _replace_entities(self, text, sentence):
        """Rebuild the text, swapping each confident entity span for its <TAG>."""
        spans = [
            label for label in sentence.get_labels()
            if label.value in self.tags and label.score > self.min_confidence
        ]
        spans.sort(key=lambda label: label.data_point.start_position)

        parts, cursor = [], 0
        for label in spans:
            parts.append(text[cursor:label.data_point.start_position])
            parts.append(f"<{label.value}>")
            cursor = label.data_point.end_position
        parts.append(text[cursor:])  # the notebook used [cursor:-1], which drops the last character
        return "".join(parts)
