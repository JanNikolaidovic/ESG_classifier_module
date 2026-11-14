"""Download the NLTK data used by chunking and masking on first use."""

from functools import cache

import nltk

RESOURCES = [("stopwords", "corpora/stopwords"), ("punkt_tab", "tokenizers/punkt_tab")]


@cache  # check once per process
def ensure_nltk_data():
    for resource, path in RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
