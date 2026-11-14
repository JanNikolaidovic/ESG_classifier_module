from esg_screening.esgify.chunking import make_windows, split_sentences


def test_split_sentences():
    text = "The plant uses water. It employs 120 people!  Is it safe?"
    assert split_sentences(text) == ["The plant uses water.", "It employs 120 people!", "Is it safe?"]


def test_split_sentences_empty_text():
    assert split_sentences("   \n ") == []


def test_windows_overlap_with_stride_one():
    assert make_windows(5, 3) == [(0, 3), (1, 4), (2, 5)]


def test_short_text_becomes_one_window():
    assert make_windows(2, 3) == [(0, 2)]
    assert make_windows(3, 3) == [(0, 3)]


def test_no_sentences_no_windows():
    assert make_windows(0, 3) == []


def test_every_sentence_is_in_a_window():
    windows = make_windows(10, 3)
    covered = {i for start, end in windows for i in range(start, end)}
    assert covered == set(range(10))
