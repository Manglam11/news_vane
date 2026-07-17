"""Prove the ANALYTICS contract -- the wiring, not the maths.

B1-B3 already proved each engine's numbers. This proves only that summarise()
returns the four frozen keys and routes real data into each one. One seeded
week feeds all three engines at once; drift stays None until Phase 6.
"""

from datetime import UTC, datetime, timedelta

import pytest

from newsvane.analytics.summary import summarise
from newsvane.storage import repository


@pytest.fixture
def seeded_week():
    # World climbs across the week and then spikes on the last day, so trends,
    # distribution and anomalies all have something real to report at once.
    days = [datetime(2026, 7, 1, 9, 0, tzinfo=UTC) + timedelta(days=i) for i in range(7)]
    baseline = [2, 3, 2, 3, 2, 3]

    articles = []
    for day, n in zip(days[:-1], baseline, strict=True):
        articles += [{"text": f"world {day} {i}", "topic": "World", "timestamp": day} for i in range(n)]
        articles += [{"text": f"sports {day} {i}", "topic": "Sports", "timestamp": day} for i in range(2)]
    articles += [{"text": f"world spike {i}", "topic": "World", "timestamp": days[-1]} for i in range(20)]

    repository.save_articles(articles)
    return days[0], days[-1] + timedelta(days=1)


def test_summarise_returns_the_four_contract_keys(seeded_week):
    start, end = seeded_week
    pulse = summarise(start, end)

    # The frozen contract: exactly these four keys, no more, no less.
    assert set(pulse) == {"trends", "distribution", "anomalies", "drift"}


def test_summarise_routes_data_into_each_engine(seeded_week):
    start, end = seeded_week
    pulse = summarise(start, end)

    # trends carries a per-topic series; World is present and climbing.
    assert "World" in pulse["trends"]

    # distribution compared the last day to the norm and returned a real result.
    assert pulse["distribution"] is not None
    assert "distance" in pulse["distribution"]

    # anomalies caught the last-day World spike.
    assert any(record["topic"] == "World" for record in pulse["anomalies"])

    # drift now fires: the seeded week is skewed far from the balanced training mix.
    assert pulse["drift"] is not None
    assert pulse["drift"]["is_drifting"] is True