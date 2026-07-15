"""Volume spikes -- did one topic jump far outside its normal range?

B2 watched the whole day's SHAPE. This zooms into a single topic's volume and
asks: is today's article-count wildly unusual for THIS topic? I answer with a
z-score -- how many standard deviations today sits from the topic's own recent
average. A z of 4 means "four wobbles above normal", which essentially never
happens by chance, so it is a real breaking-event signal.

I build the baseline from the days BEFORE today, never including today. If
today's spike were folded into its own average it would drag that average up
and partly hide itself -- the same time-honest instinct as the half-open window.
"""

from datetime import datetime
from statistics import mean, stdev

from config.settings import settings
from newsvane.storage.repository import count_by_day

TOPICS = list(settings.scraper_sections)

# How many standard deviations from normal before I call it a spike. A setting,
# not a magic number -- tightening the radar's sensitivity is a config edit.
DEFAULT_Z_THRESHOLD = settings.anomaly_z_threshold


def _z_score(today: int, history: list[int]) -> float | None:
    # How many standard deviations today sits above (or below) the historical
    # mean. Needs at least two prior days to have a spread at all; with fewer,
    # or when the topic was dead flat (stdev 0), "unusual" has no meaning yet.
    if len(history) < 2:
        return None
    spread = stdev(history)
    if spread == 0:
        return None
    return (today - mean(history)) / spread


def _series_by_topic(rows: list[dict]) -> dict[str, dict[datetime, int]]:
    # Reshape flat (day, topic, count) rows into per-topic day->count maps.
    series: dict[str, dict[datetime, int]] = {t: {} for t in TOPICS}
    for row in rows:
        series[row["topic"]][row["day"]] = row["count"]
    return series


def volume_anomalies(
    start: datetime,
    end: datetime,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> list[dict]:
    """Flag topics whose LAST-day volume is an outlier versus their own history.

    For each topic I take its counts across the window, hold back the final day
    as "today", and z-score today against every day before it. A topic clears
    the bar only when its z-score meets the threshold. Returns one record per
    flagged topic, busiest anomaly first. An empty list means a calm news day.
    """
    series = _series_by_topic(count_by_day(start, end))

    all_days = sorted({day for counts in series.values() for day in counts})
    if len(all_days) < 3:  # need >=2 history days + today for any spread
        return []

    today = all_days[-1]
    history_days = all_days[:-1]

    flagged = []
    for topic, counts in series.items():
        today_count = counts.get(today, 0)
        history = [counts.get(day, 0) for day in history_days]

        z = _z_score(today_count, history)
        if z is not None and z >= z_threshold:
            flagged.append(
                {
                    "topic": topic,
                    "day": today,
                    "count": today_count,
                    "baseline": mean(history),
                    "z_score": z,
                }
            )

    return sorted(flagged, key=lambda record: record["z_score"], reverse=True)