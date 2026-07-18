"""Prove the drift alarm -- the live window versus the model's training mix.

topic_mix_drift reuses the very Jensen-Shannon distance the distribution tests
already prove, so here I test only the DRIFT-specific behaviour: an empty window
stays silent, a window whose mix matches the training reference does not fire,
and a window skewed far from it does. The reference is the frozen 25%-per-class
training shape from settings.
"""

from datetime import UTC, datetime, timedelta

import pytest

from newsvane.analytics.drift import topic_mix_drift
from newsvane.storage import repository


def test_empty_window_returns_none():
    # No articles collected means no live shape to compare -- silence is the
    # honest answer, not a distance invented from nothing.
    start = datetime(2026, 7, 1, tzinfo=UTC)
    assert topic_mix_drift(start, start + timedelta(days=1)) is None


def test_matching_mix_does_not_drift():
    # One article per class -- a live mix of exactly 25% each, identical to the
    # training reference. Distance is 0, so the alarm stays silent.
    day = datetime(2026, 7, 2, 9, 0, tzinfo=UTC)
    repository.save_articles(
        [
            {"text": "world one", "topic": "World", "timestamp": day},
            {"text": "sports one", "topic": "Sports", "timestamp": day},
            {"text": "business one", "topic": "Business", "timestamp": day},
            {"text": "scitech one", "topic": "Sci/Tech", "timestamp": day},
        ]
    )

    result = topic_mix_drift(day, day + timedelta(days=1))

    assert result is not None
    assert result["distance"] == pytest.approx(0.0)
    assert result["is_drifting"] is False


def test_skewed_mix_raises_the_alarm():
    # A window of nothing but World is as far from a balanced training mix as the
    # data gets -- the distance must clear the threshold and fire the alarm.
    day = datetime(2026, 7, 3, 9, 0, tzinfo=UTC)
    repository.save_articles(
        [{"text": f"world story {i}", "topic": "World", "timestamp": day} for i in range(5)]
    )

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
