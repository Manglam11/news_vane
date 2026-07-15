"""Prove the topic-mix maths -- pure divergence first, then end-to-end.

I test in two layers. First the pure functions on hand-computed vectors, where
I know the exact answer and no database is involved -- identical shapes must
give 0, opposite shapes must give 1. Only then a seeded-week check that the
whole topic_mix_shift pipeline reshapes real rows correctly. If the pure layer
is right and the seeded layer is right, the maths is trustworthy.
"""

from datetime import UTC, datetime, timedelta

import pytest

from newsvane.analytics.distributions import (
    _js_divergence,
    topic_mix_shift,
)
from newsvane.storage import repository


def test_identical_shapes_have_zero_distance():
    # Two identical distributions are zero apart -- the defining property.
    p = [0.25, 0.25, 0.25, 0.25]
    assert _js_divergence(p, p) == 0.0


def test_opposite_shapes_have_max_distance():
    # Two distributions with no overlap sit at the ceiling: base-2 JS = 1.0.
    p = [1.0, 0.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0, 0.0]
    assert _js_divergence(p, q) == pytest.approx(1.0)


def test_partial_overlap_sits_between():
    # A shape that half-agrees lands strictly between the two extremes.
    p = [0.5, 0.5, 0.0, 0.0]
    q = [0.5, 0.0, 0.5, 0.0]
    distance = _js_divergence(p, q)
    assert 0.0 < distance < 1.0


def test_too_little_history_returns_none():
    # One day alone has nothing to compare against -- the function must say so
    # rather than inventing a distance from a single day.
    day = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    repository.save_articles([{"text": "only day", "topic": "World", "timestamp": day}])
    assert topic_mix_shift(day, day + timedelta(days=1)) is None


def test_topic_mix_shift_flags_a_lurch(seeded_lurch):
    # Two calm days of pure World, then a day that lurches to pure Sports. The
    # distance from that last day to the norm should be large -- near the ceiling.
    start, end = seeded_lurch
    result = topic_mix_shift(start, end)

    assert result is not None
    assert result["today"]["Sports"] == pytest.approx(1.0)
    assert result["distance"] > 0.9


@pytest.fixture
def seeded_lurch():
    # Days 1-2: entirely World. Day 3: entirely Sports. A deliberate, extreme
    # shape change, so a correct distance must come back large.
    day1 = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)

    articles = [
        {"text": "world a", "topic": "World", "timestamp": day1},
        {"text": "world b", "topic": "World", "timestamp": day2},
        {"text": "sports a", "topic": "Sports", "timestamp": day3},
    ]
    repository.save_articles(articles)

    return day1, day3 + timedelta(days=1)