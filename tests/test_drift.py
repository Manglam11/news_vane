"""Prove the drift alarm -- the MODEL's live output versus its training mix.

topic_mix_drift reuses the very Jensen-Shannon distance the distribution tests
already prove, so here I test only the DRIFT-specific behaviour: a window with no
classified rows stays silent, a predicted mix matching the training reference does
not fire, and one skewed far from it does.

The sharpest test in this file is the one where the SECTIONS are balanced and the
model's answers are not. Drift must fire on that, because the mix it measures is
the model's, not my scraper's quota. If anyone ever points this back at topic,
that test goes red and says why.
"""

from datetime import UTC, datetime, timedelta

import pytest

from newsvane.analytics.drift import topic_mix_drift
from newsvane.storage import repository


def article(text: str, topic: str, day: datetime, predicted: str | None = None) -> dict:
    # Every row here carries both answers: the section it came from and what the
    # model said. predicted=None is a row the model was never asked about.
    return {
        "text": text,
        "topic": topic,
        "timestamp": day,
        "predicted_label": predicted,
        "predicted_score": None if predicted is None else 0.9,
    }


def test_empty_window_returns_none():
    # No articles collected means no live shape to compare -- silence is the
    # honest answer, not a distance invented from nothing.
    start = datetime(2026, 7, 1, tzinfo=UTC)
    assert topic_mix_drift(start, start + timedelta(days=1)) is None


def test_unclassified_rows_return_none():
    # The window is full, and every row predates the model being wired into the
    # harvest. A prediction that was never made is not a vote for any class, so
    # drift must stay silent rather than report a mix built from nothing.
    day = datetime(2026, 7, 4, 9, 0, tzinfo=UTC)
    repository.save_articles(
        [article(f"old story {i}", "World", day) for i in range(5)],
    )

    assert topic_mix_drift(day, day + timedelta(days=1)) is None


def test_matching_mix_does_not_drift():
    # One article per class, and the model agreed with every one -- a live mix of
    # exactly 25% each, identical to the training reference. Distance is 0.
    day = datetime(2026, 7, 2, 9, 0, tzinfo=UTC)
    repository.save_articles(
        [
            article("world one", "World", day, "World"),
            article("sports one", "Sports", day, "Sports"),
            article("business one", "Business", day, "Business"),
            article("scitech one", "Sci/Tech", day, "Sci/Tech"),
        ]
    )

    result = topic_mix_drift(day, day + timedelta(days=1))

    assert result is not None
    assert result["distance"] == pytest.approx(0.0)
    assert result["is_drifting"] is False
    assert result["agreement"] == {"agreed": 4, "scored": 4, "rate": pytest.approx(1.0)}


def test_skewed_mix_raises_the_alarm():
    # A window where the model answered nothing but World is as far from a balanced
    # training mix as the data gets -- the distance must clear the threshold.
    day = datetime(2026, 7, 3, 9, 0, tzinfo=UTC)
    repository.save_articles([article(f"world story {i}", "World", day, "World") for i in range(5)])

    result = topic_mix_drift(day, day + timedelta(days=1))

    assert result is not None
    assert result["distance"] > 0.5
    assert result["is_drifting"] is True
    assert result["reference"] == {
        "World": 0.25,
        "Sports": 0.25,
        "Business": 0.25,
        "Sci/Tech": 0.25,
    }


def test_drift_measures_the_model_not_the_sections():
    # The scraper caps every section at the same number, so a mix built from
    # sections is mostly a picture of my own quota. Here the sections are perfectly
    # balanced -- one each -- and the model called all four World. Drift must fire
    # on the model's answer. Grouped by topic instead, this window is a flat 25%
    # and the distance would be zero.
    day = datetime(2026, 7, 5, 9, 0, tzinfo=UTC)
    repository.save_articles(
        [
            article("world one", "World", day, "World"),
            article("sports one", "Sports", day, "World"),
            article("business one", "Business", day, "World"),
            article("scitech one", "Sci/Tech", day, "World"),
        ]
    )

    result = topic_mix_drift(day, day + timedelta(days=1))

    assert result is not None
    assert result["is_drifting"] is True
    assert result["live"]["World"] == pytest.approx(1.0)
    # One right out of four, and the mark is the evidence of the disagreement.
    assert result["agreement"] == {"agreed": 1, "scored": 4, "rate": pytest.approx(0.25)}


def test_agreement_ignores_rows_the_model_never_answered():
    # A model that fails to answer must shrink the exam, never lower the mark.
    # Counting an unasked question as wrong makes an outage look like a mistake.
    day = datetime(2026, 7, 6, 9, 0, tzinfo=UTC)
    repository.save_articles(
        [
            article("world one", "World", day, "World"),
            article("world two", "World", day, "World"),
            article("world three", "World", day),
            article("world four", "World", day),
        ]
    )

    result = topic_mix_drift(day, day + timedelta(days=1))

    assert result is not None
    assert result["agreement"] == {"agreed": 2, "scored": 2, "rate": pytest.approx(1.0)}
