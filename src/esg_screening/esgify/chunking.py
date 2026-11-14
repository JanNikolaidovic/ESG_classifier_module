"""Split long project descriptions into chunks the ESGify model can score.

Scoring the whole description at once dilutes the signal (the model averages all tokens
into one vector), while single sentences lack context and give false alarms. Overlapping
windows of a few consecutive sentences work best on our samples; see the README.
"""

from nltk.tokenize import sent_tokenize

from esg_screening.esgify.nltk_data import ensure_nltk_data


def split_sentences(text):
    """Split text into sentences, dropping empty ones."""
    ensure_nltk_data()
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def make_windows(n_sentences, window_size):
    """Return (start, end) sentence index ranges for overlapping windows with stride 1.

    Example: 5 sentences, window_size 3 -> (0, 3), (1, 4), (2, 5).
    Texts shorter than one window become a single chunk.
    """
    if n_sentences == 0:
        return []
    if n_sentences <= window_size:
        return [(0, n_sentences)]
    return [(start, start + window_size) for start in range(n_sentences - window_size + 1)]
