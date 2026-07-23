"""Pure: the API's JSON turned into the tables the panels draw.

Nothing here imports streamlit or httpx, and that is deliberate. Those two live
in a dependency group CI does not install, so a test that had to import them
would turn the whole suite red on the runner. Keeping the maths in a module with
no framework in it is what lets this box be tested at all.
"""

import pandas as pd

# Below this many marked articles I refuse to read the drift number out loud. A
# four-class divergence over fifteen rows moves wildly on a single article: it is
# mechanically correct and statistically empty, and a dashboard that prints it as
# a verdict is the most confident kind of wrong.
MIN_SCORED = 30


def momentum_frame(trends: dict) -> pd.DataFrame:
    """The ragged per-topic series, lined up on one shared daily axis.

    The API returns only the days a topic actually had articles on, so the series
    arrive different lengths. Drawn raw, a topic that harvested nothing on Tuesday
    gets its line jumped straight over the gap -- which reads as "no data" when the
    truth is zero. A day inside the window with no articles IS a zero, and drawing
    it as anything else is the same lie a total that hides a zero tells.
    """
    points = [
        {"day": point["day"], "topic": topic, "count": point["count"]}
        for topic, series in trends.items()
        for point in series
    ]
    if not points:
        return pd.DataFrame()

    frame = pd.DataFrame(points)
    frame["day"] = pd.to_datetime(frame["day"]).dt.normalize()

    wide = frame.pivot(index="day", columns="topic", values="count")
    every_day = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(every_day).fillna(0).astype(int)
    wide.index.name = "day"

    return wide.reset_index().melt(id_vars="day", var_name="topic", value_name="count")


def mix_frame(distribution: dict) -> pd.DataFrame:
    """Today's topic-mix beside the recent norm, as long rows for one grouped chart.

    The topic list is seeded from BOTH mixes, never from one of them. A topic that
    held a share of the norm and none of today is the most interesting bar on the
    chart, and building the rows from today's keys alone would drop it silently --
    the same disease as a count built only from what arrived.
    """
    today = distribution["today"]
    norm = distribution["norm"]

    rows = []
    for topic in sorted(set(today) | set(norm)):
        # An absent topic is a share of zero, which is a reading, not a gap.
        rows.append({"topic": topic, "when": "today", "share": today.get(topic, 0.0)})
        rows.append({"topic": topic, "when": "recent norm", "share": norm.get(topic, 0.0)})

    return pd.DataFrame(rows)


def drift_verdict(drift: dict | None) -> tuple[str, str]:
    """Turn the drift block into (verdict, why) -- a sentence, not a float.

    0.0496 tells a reader nothing on its own. It has to be said against the line
    it is being compared to, and against the number of articles it was computed
    from, or it is a decimal pretending to be an answer.
    """
    if drift is None:
        return "no reading", "no article in this window carries a prediction yet"

    distance = drift["distance"]
    threshold = drift["threshold"]
    scored = drift["agreement"]["scored"]

    if scored < MIN_SCORED:
        return "too few marked", f"only {scored} articles marked -- one article moves this number"
    if drift["is_drifting"]:
        return "drifting", f"{distance:.4f}, past the {threshold} line"
    return "steady", f"{distance:.4f}, under the {threshold} line"


def anomaly_frame(anomalies: list[dict]) -> pd.DataFrame:
    """The breakout topics, loudest first.

    An empty list is the normal state of this table, not a missing reading -- most
    days nothing spikes. The panel says so in words rather than drawing an empty
    grid, because a blank table reads like a broken query.
    """
    if not anomalies:
        return pd.DataFrame()

    frame = pd.DataFrame(anomalies)
    frame["day"] = pd.to_datetime(frame["day"]).dt.strftime("%b %d")
    frame = frame.reindex(frame["z_score"].abs().sort_values(ascending=False).index)

    return frame[["topic", "day", "count", "baseline", "z_score"]].reset_index(drop=True)
