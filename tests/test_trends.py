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


@pytest.fixture
def seeded_moods():
    # Three days built to separate the two numbers that share a row. Day one is
    # fully read, day two is half read, day three was never read at all -- and
    # the count must be blind to all of that while the mood must not.
    day1 = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)

    articles = [
        {"text": "mood a", "topic": "World", "timestamp": day1, "sentiment_score": 0.6},
        {"text": "mood b", "topic": "World", "timestamp": day1, "sentiment_score": 0.2},
        {"text": "mood c", "topic": "World", "timestamp": day2, "sentiment_score": 0.5},
        # No sentiment keys at all -- the bare DATA contract, exactly as every
        # row written before the mood engine shipped looks in the table today.
        {"text": "mood d", "topic": "World", "timestamp": day2},
        {"text": "mood e", "topic": "World", "timestamp": day3},
    ]
    repository.save_articles(articles)

    return day1, day3 + timedelta(days=1)


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


def test_the_atom_carries_a_mood_beside_the_count(seeded_moods):
    # The blueprint always specified this atom as "article-count AND average
    # sentiment day by day". Only half of it existed for nineteen sessions.
    start, end = seeded_moods
    first_day = topic_momentum(start, end)["World"][0]

    assert set(first_day) == {"day", "count", "mood"}
    assert first_day["count"] == 2
    assert first_day["mood"] == pytest.approx(0.4)


def test_the_two_numbers_have_two_denominators(seeded_moods):
    # The keystone. Day two holds two articles and exactly one reading. The
    # count must say 2 because two articles were published; the mood must
    # average over 1, because averaging an unread article in as a zero would
    # drag a genuinely positive day towards neutral.
    start, end = seeded_moods
    second_day = topic_momentum(start, end)["World"][1]

    assert second_day["count"] == 2
    assert second_day["mood"] == pytest.approx(0.5)


def test_a_day_nobody_read_reports_no_mood_and_still_reports_its_count(seeded_moods):
    # NULL is not 0.0. A day the mood engine never saw must come back as None so
    # the dashboard can leave a break in the line, rather than as a zero that
    # draws an engine outage as a calm news day.
    start, end = seeded_moods
    third_day = topic_momentum(start, end)["World"][2]

    assert third_day["count"] == 1
    assert third_day["mood"] is None
