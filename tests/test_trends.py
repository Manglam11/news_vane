"""Prove the momentum maths on a seeded fake week.

Real history does not exist yet -- the scraper has run once. So I manufacture
a known week of articles, insert them into the real test database, and check
that topic_momentum reshapes them into the exact per-topic daily series I
expect. The maths is proven now; the real data just fills in behind it.
"""

from datetime import UTC, datetime, timedelta

import pytest

from newsvane.analytics.trends import topic_momentum
from newsvane.storage import repository


@pytest.fixture
def seeded_week():
    # A three-day window with a deliberately uneven shape, so a wrong grouping
    # cannot accidentally pass: World climbs 1 -> 2 -> 3, Sports appears on one
    # day only, Business never appears at all.
    day1 = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)

    # save_articles expects the DATA contract shape: text, topic, timestamp.
    # I seed through it rather than the ORM so the test writes through the exact
    # same door the scraper does -- and through the redirect the db fixture sets.
    articles = [
        {"text": "world a", "topic": "World", "timestamp": day1},
        {"text": "world b", "topic": "World", "timestamp": day2},
        {"text": "world c", "topic": "World", "timestamp": day2},
        {"text": "world d", "topic": "World", "timestamp": day3},
        {"text": "world e", "topic": "World", "timestamp": day3},
        {"text": "world f", "topic": "World", "timestamp": day3},
        {"text": "sports a", "topic": "Sports", "timestamp": day2},
    ]
    repository.save_articles(articles)

    return day1, day3 + timedelta(days=1)  # half-open end: include all of day3


def test_topic_momentum_counts_per_topic_per_day(seeded_week):
    start, end = seeded_week
    series = topic_momentum(start, end)

    # Only topics that actually have articles appear -- Business is absent, not zero.
    assert set(series) == {"World", "Sports"}

    # World's daily counts climb 1 -> 2 -> 3, in ascending day order.
    world_counts = [point["count"] for point in series["World"]]
    assert world_counts == [1, 2, 3]

    # Sports appears on exactly one day, with one article.
    assert len(series["Sports"]) == 1
    assert series["Sports"][0]["count"] == 1
