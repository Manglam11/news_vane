"""Topic momentum -- the first line the radar draws, and the mood beneath it.

I turn the raw per-topic, per-day numbers from STORAGE into a tidy series: for
each topic, its article-count over consecutive days, and how those articles
read. Together they answer the two questions momentum asks -- "is coverage of
this topic rising or fading?" and "is its tone turning?" -- and they are what
the dashboard plots.

Two separate repository calls rather than one wider query, because they are two
different questions over the same rows: how MANY, and how it FEELS. Only some
rows carry a mood, so they can never share a WHERE clause honestly.
"""

from datetime import datetime

from newsvane.storage.repository import count_by_day, mood_by_day


def topic_momentum(start: datetime, end: datetime) -> dict[str, list[dict]]:
    """Return each topic's daily article-count and average mood across a window.

    Output shape: {topic: [{day, count, mood}, ...]}, days in ascending order.
    STORAGE only reports (day, topic) pairs that actually have articles, so a
    day with zero coverage of a topic is simply absent -- I do NOT invent a
    zero for it here. Filling gaps is a presentation choice, and presentation
    is the dashboard's job, not the maths engine's.

    mood is None whenever no article in that bucket carried a reading -- every
    row written before the mood engine shipped, and any row it could not read.
    That is deliberately NOT zero: neutral news scores an honest 0.0, so a zero
    standing in for "never read" would make an outage look like a calm day. The
    count is always real; the mood is a reading that may not exist.
    """
    counts = count_by_day(start, end)

    # Keyed on the same (day, topic) pair STORAGE grouped by, so the two answers
    # line up exactly rather than being matched by position or by order.
    moods = {(row["day"], row["topic"]): row["mood"] for row in mood_by_day(start, end)}

    series: dict[str, list[dict]] = {}
    for row in counts:
        series.setdefault(row["topic"], []).append(
            {
                "day": row["day"],
                "count": row["count"],
                "mood": moods.get((row["day"], row["topic"])),
            }
        )
    return series
