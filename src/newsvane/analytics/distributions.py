"""Topic-mix shape -- is today's distribution normal, or has it lurched?

B1 asked "how much of each topic?". This asks a different question: on a given
day the four topics form a MIX -- a shape -- and I want to know whether today's
shape looks like the recent norm or has moved somewhere unusual.

I turn each day's raw counts into PROPORTIONS first, so a busy news day and a
quiet one compare fairly (10 of 100 and 1 of 10 are the same shape). Then I
measure the distance between today's shape and the average of the days before
it, using Jensen-Shannon divergence: a standard, symmetric, bounded [0, 1]
distance between two probability distributions. The very same number becomes
the drift alarm in Phase 6 -- one statistic, two jobs.
"""

from datetime import datetime
from math import log2

from config.settings import settings

from newsvane.storage.repository import count_by_day

# The topic order is fixed and tied to the model's label space, so every day's
# proportion vector lines up slot-for-slot. A topic absent on a day is a real
# zero in its slot, not a missing entry.
TOPICS = list(settings.scraper_sections)


def _daily_proportions(rows: list[dict]) -> dict[datetime, list[float]]:
    # Reshape the flat (day, topic, count) rows into one proportion vector per
    # day, each vector summing to 1 across the fixed TOPICS order.
    counts: dict[datetime, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["day"], {})[row["topic"]] = row["count"]

    proportions: dict[datetime, list[float]] = {}
    for day, topic_counts in counts.items():
        total = sum(topic_counts.values())
        proportions[day] = [topic_counts.get(t, 0) / total for t in TOPICS]
    return proportions


def _kl(p: list[float], q: list[float]) -> float:
    # Kullback-Leibler divergence, base 2. Where p is 0 the term is 0 (the limit
    # of x*log(x) as x->0), so an absent topic contributes nothing rather than
    # exploding. q is never 0 here because I only ever feed it the mixture M.
    return sum(pi * log2(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0)


def js_divergence(p: list[float], q: list[float]) -> float:
    # Jensen-Shannon: the symmetric, always-finite average of each side's KL to
    # their midpoint M. Bounded in [0, 1] with base-2 logs. 0 = identical shapes,
    # 1 = maximally different. Public because the drift alarm reuses it verbatim --
    # one statistic, two jobs, and I refuse to keep two copies of one formula.
    m = [(pi + qi) / 2 for pi, qi in zip(p, q, strict=True)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def topic_mix_shift(start: datetime, end: datetime) -> dict | None:
    """Compare the LAST day's topic-mix to the average of the days before it.

    Returns {day, distance, today, norm} where distance is the JS divergence in
    [0, 1], today is the last day's proportions, and norm is the averaged
    proportions of every prior day. Returns None when there is not enough
    history -- I need at least one reference day plus today to compare at all.
    """
    proportions = _daily_proportions(count_by_day(start, end))
    if len(proportions) < 2:
        return None

    days = sorted(proportions)
    today = proportions[days[-1]]
    prior = [proportions[d] for d in days[:-1]]

    # The "norm" is the element-wise mean shape across every prior day.
    norm = [sum(vec[i] for vec in prior) / len(prior) for i in range(len(TOPICS))]

    return {
        "day": days[-1],
        "distance": js_divergence(today, norm),
        "today": dict(zip(TOPICS, today, strict=True)),
        "norm": dict(zip(TOPICS, norm, strict=True)),
    }
