"""Prove the spike maths -- pure z-score first, then end-to-end.

Same two-layer discipline as the distribution tests. First the z-score on
hand-picked numbers where I know the answer: a flat history has no spread so
no spike is possible, and a value far above a steady baseline scores high.
Then a seeded week with one deliberate 5x spike that must get flagged, and a
calm week that must stay silent.
"""

from datetime import UTC, datetime, timedelta

import pytest

from newsvane.analytics.anomalies import _z_score, volume_anomalies
from newsvane.storage import repository


def test_flat_history_has_no_z_score():
    # A topic that ran dead flat has zero spread -- "unusual" is undefined, so
    # the function must return None rather than dividing by zero.
    assert _z_score(10, [5, 5, 5, 5]) is None


def test_single_history_day_has_no_z_score():
    # One prior day is not enough to know what "normal wobble" even is.
    assert _z_score(10, [5]) is None


def test_value_far_above_baseline_scores_high():
    # A steady baseline of ~2 with a jump to 20 is many standard deviations out.
    z = _z_score(20, [2, 2, 3, 2, 1])
    assert z is not None
    assert z > 3.0


def _spread_days(start: datetime, count: int) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(count)]


def test_volume_anomalies_flags_a_spike(seeded_spike):
    # Sports sits at 2/day for a week, then jumps to 20. It must be flagged,
    # and with only Sports spiking it should be the sole record.
    start, end = seeded_spike
    flagged = volume_anomalies(start, end)

    assert len(flagged) == 1
    assert flagged[0]["topic"] == "Sports"
    assert flagged[0]["count"] == 20
    assert flagged[0]["z_score"] > 3.0


def test_calm_week_flags_nothing(seeded_calm):
    # Every topic steady around its own level -- no spike, an empty list.
    start, end = seeded_calm
    assert volume_anomalies(start, end) == []


@pytest.fixture
def seeded_spike():
    # Six days of Sports wobbling gently around ~2 (never dead flat -- a real
    # topic has some spread, and a zero-spread baseline makes a z-score
    # undefined), then a seventh day at 20: a clean spike towering over the norm.
    days = _spread_days(datetime(2026, 7, 1, 9, 0, tzinfo=UTC), 7)
    baseline_counts = [2, 3, 1, 2, 3, 2]  # gentle wobble, mean ~2.2, real spread

    articles = []
    for day, n in zip(days[:-1], baseline_counts, strict=True):
        articles += [
            {"text": f"sports {day} {i}", "topic": "Sports", "timestamp": day} for i in range(n)
        ]
    articles += [
        {"text": f"sports spike {i}", "topic": "Sports", "timestamp": days[-1]} for i in range(20)
    ]

    repository.save_articles(articles)
    return days[0], days[-1] + timedelta(days=1)


@pytest.fixture
def seeded_calm():
    # World steady at 3/day across seven days -- normal wobble is zero, nothing
    # to flag. Proves the detector does not cry wolf on a quiet stream.
    days = _spread_days(datetime(2026, 8, 1, 9, 0, tzinfo=UTC), 7)

    articles = []
    for day in days:
        articles += [
            {"text": f"world {day} {i}", "topic": "World", "timestamp": day} for i in range(3)
        ]

    repository.save_articles(articles)
    return days[0], days[-1] + timedelta(days=1)
