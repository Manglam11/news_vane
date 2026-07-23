"""The DASHBOARD box's maths, tested without ever starting a browser.

Every function here comes from shaping.py, which imports neither streamlit nor
httpx. That is what makes this file runnable in CI: the chart libraries live in a
dependency group the runner never installs, so a test that reached for a panel
would turn the whole suite red on a machine that is otherwise perfectly healthy.
"""

import pandas as pd

from newsvane.dashboard.shaping import (
    MIN_SCORED,
    anomaly_frame,
    drift_verdict,
    mix_frame,
    momentum_frame,
)


def test_momentum_draws_a_missing_day_as_zero():
    # The keystone. The API only sends the days a topic had articles on, so a
    # silent day arrives as an absent row -- and an absent row drawn on a chart
    # reads as "no data" when the truth is "nothing happened".
    trends = {
        "World": [
            {"day": "2026-07-20T00:00:00Z", "count": 5},
            {"day": "2026-07-22T00:00:00Z", "count": 7},
        ]
    }

    frame = momentum_frame(trends)
    counts = frame.set_index("day")["count"]

    assert len(frame) == 3
    assert counts[pd.Timestamp("2026-07-21", tz="UTC")] == 0


def test_momentum_starts_a_late_topic_at_zero_not_at_nothing():
    # Sports had no rows for the first day of the window. Its line must begin at
    # the same date as everyone else's, sitting on zero.
    trends = {
        "World": [
            {"day": "2026-07-20T00:00:00Z", "count": 5},
            {"day": "2026-07-21T00:00:00Z", "count": 6},
        ],
        "Sports": [{"day": "2026-07-21T00:00:00Z", "count": 3}],
    }

    frame = momentum_frame(trends)
    sports = frame[frame["topic"] == "Sports"].set_index("day")["count"]

    assert len(sports) == 2
    assert sports[pd.Timestamp("2026-07-20", tz="UTC")] == 0


def test_momentum_survives_an_empty_window():
    assert momentum_frame({}).empty


def test_mix_keeps_a_topic_that_vanished_today():
    # A topic that held a third of the recent norm and none of today is the most
    # interesting bar on the chart. Seeding from today's keys alone would drop it.
    distribution = {
        "today": {"World": 1.0},
        "norm": {"World": 0.7, "Sports": 0.3},
    }

    frame = mix_frame(distribution)
    sports_today = frame[(frame["topic"] == "Sports") & (frame["when"] == "today")]

    assert set(frame["topic"]) == {"World", "Sports"}
    assert sports_today["share"].item() == 0.0


def test_drift_verdict_refuses_to_read_a_tiny_sample():
    # is_drifting is False here, and the honest answer is still not "steady".
    # A four-class divergence over a handful of rows moves on one article.
    drift = {
        "distance": 0.05,
        "threshold": 0.1,
        "is_drifting": False,
        "agreement": {"agreed": 13, "scored": MIN_SCORED - 1, "rate": 0.867},
    }

    verdict, why = drift_verdict(drift)

    assert verdict == "too few marked"
    assert str(MIN_SCORED - 1) in why


def test_drift_verdict_reads_steady_once_the_sample_is_big_enough():
    drift = {
        "distance": 0.05,
        "threshold": 0.1,
        "is_drifting": False,
        "agreement": {"agreed": 90, "scored": MIN_SCORED, "rate": 0.9},
    }

    assert drift_verdict(drift)[0] == "steady"


def test_drift_verdict_fires_when_the_model_has_moved():
    drift = {
        "distance": 0.42,
        "threshold": 0.1,
        "is_drifting": True,
        "agreement": {"agreed": 50, "scored": MIN_SCORED, "rate": 0.5},
    }

    verdict, why = drift_verdict(drift)

    assert verdict == "drifting"
    assert "0.1" in why


def test_drift_verdict_handles_a_window_with_no_predictions():
    # /pulse types drift as DriftOut | None. The page must survive the None.
    assert drift_verdict(None)[0] == "no reading"


def test_anomalies_are_empty_on_a_normal_day():
    # An empty list is this table's usual state, not a failed query.
    assert anomaly_frame([]).empty


def test_anomalies_are_ordered_loudest_first():
    # A collapse is as newsworthy as a spike, so the sort is on the SIZE of the
    # z-score. Ordering on the raw value would bury every negative one at the end.
    anomalies = [
        {
            "topic": "World",
            "day": "2026-07-23T00:00:00Z",
            "count": 20,
            "baseline": 12.0,
            "z_score": 2.0,
        },
        {
            "topic": "Sports",
            "day": "2026-07-23T00:00:00Z",
            "count": 0,
            "baseline": 8.0,
            "z_score": -4.5,
        },
        {
            "topic": "Business",
            "day": "2026-07-23T00:00:00Z",
            "count": 30,
            "baseline": 10.0,
            "z_score": 3.1,
        },
    ]

    frame = anomaly_frame(anomalies)

    assert list(frame["topic"]) == ["Sports", "Business", "World"]
