"""Prove the mood engine's own decisions, not VADER's dictionary.

Two things are mine here and nothing else is: where I cut the compound score
into three words, and the promise that both halves of the reading come back.
The lexicon itself is the library's business -- but its GRAMMAR rules are the
whole reason I chose it over counting positive words, so one test holds them
to that.
"""

from config.settings import settings

from newsvane.models.sentiment import label_for, load_analyzer, read_sentiment


def test_both_boundaries_count_towards_the_pole():
    # The decision, pinned exactly like the scraper's 72-hour gate: a score
    # sitting precisely ON a threshold belongs to the pole, not to the neutral
    # band. Flip either comparison to a strict one and this goes red.
    assert label_for(settings.sentiment_positive_threshold) == "positive"
    assert label_for(settings.sentiment_negative_threshold) == "negative"


def test_just_inside_the_band_is_neutral():
    assert label_for(settings.sentiment_positive_threshold - 0.01) == "neutral"
    assert label_for(settings.sentiment_negative_threshold + 0.01) == "neutral"
    # Dead centre. Flat factual coverage is a real reading, not a missing one.
    assert label_for(0.0) == "neutral"


def test_negation_flips_the_sign():
    # The reason a lexicon was an honest choice at all. A bag of positive words
    # scores these two identically -- both contain "good" -- and a mood engine
    # that cannot read "not" would report the news backwards on any headline
    # about a failure. Same words, opposite verdicts.
    _, upbeat = read_sentiment("This is a good result for the team")
    _, bleak = read_sentiment("This is not a good result for the team")

    assert upbeat > 0
    assert bleak < 0


def test_read_sentiment_hands_back_a_word_and_a_number():
    # Two halves answering two questions: the word rides the frozen MODEL
    # contract to the dashboard, the number is what ANALYTICS can average.
    # You cannot take the mean of three adjectives.
    label, compound = read_sentiment("A stunning comeback delighted the crowd")

    assert isinstance(label, str)
    assert isinstance(compound, float)
    assert -1.0 <= compound <= 1.0
    # The two halves must never disagree about the same article.
    assert label == label_for(compound)


def test_the_analyzer_is_built_once():
    # Building it parses a 434 KB lexicon off disk. Paying that per article
    # would cost far more than the scoring itself, so the cache is load-bearing
    # rather than a nicety.
    assert load_analyzer() is load_analyzer()
