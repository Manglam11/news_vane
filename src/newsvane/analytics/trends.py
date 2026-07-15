"""Topic momentum -- the first line the radar draws.

I turn the raw per-topic, per-day counts from STORAGE into a tidy series:
for each topic, its article-count over consecutive days. That series is the
answer to the only question momentum asks -- "is coverage of this topic
rising or fading?" -- and it is what the dashboard plots.
"""

from datetime import datetime

from newsvane.storage.repository import count_by_day


def topic_momentum(start: datetime, end: datetime) -> dict[str, list[dict]]:
    """Return each topic's daily article-count across a half-open window.

    Output shape: {topic: [{day, count}, ...]}, days in ascending order.
    STORAGE only reports (day, topic) pairs that actually have articles, so a
    day with zero coverage of a topic is simply absent -- I do NOT invent a
    zero for it here. Filling gaps is a presentation choice, and presentation
    is the dashboard's job, not the maths engine's.
    """
    rows = count_by_day(start, end)

    series: dict[str, list[dict]] = {}
    for row in rows:
        series.setdefault(row["topic"], []).append(
            {"day": row["day"], "count": row["count"]}
        )
    return series