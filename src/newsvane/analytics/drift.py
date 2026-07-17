"""Has the live news drifted away from what the model was trained on?

topic_mix_shift asks "is TODAY unusual versus recent days?". Drift asks the
production question: "has the whole recent stream moved away from the training
distribution, so the model is quietly going stale?". Same maths, different
yardstick -- there I compared today to its own recent norm; here I compare the
recent window to the FROZEN training mix (settings.drift_reference).

I aggregate every article in the window into one live topic-mix, then measure
its Jensen-Shannon distance from the training mix. Above the threshold, the
radar says the ground the model stands on has moved. This is the fourth key the
summarise() contract has held open as None since Phase 4 -- one statistic doing
its second job, exactly as the blueprint promised.
"""

from collections import Counter
from datetime import datetime

from config.settings import settings
from newsvane.analytics.distributions import TOPICS, js_divergence
from newsvane.storage.repository import count_by_day


def topic_mix_drift(start: datetime, end: datetime) -> dict | None:
    """Measure how far the window's topic-mix has drifted from training.

    Returns {distance, threshold, is_drifting, live, reference}. distance is the
    JS divergence in [0, 1] between the live window's mix and the frozen training
    mix; is_drifting is that distance crossing settings.drift_threshold. Returns
    None when the window holds no articles -- with nothing collected, there is no
    live shape to compare and silence is the honest answer.
    """
    counts: Counter[str] = Counter()
    for row in count_by_day(start, end):
        counts[row["topic"]] += row["count"]

    total = sum(counts.values())
    if total == 0:
        return None

    live = [counts[t] / total for t in TOPICS]
    reference = [settings.drift_reference[t] for t in TOPICS]
    distance = js_divergence(live, reference)

    return {
        "distance": distance,
        "threshold": settings.drift_threshold,
        "is_drifting": distance > settings.drift_threshold,
        "live": dict(zip(TOPICS, live, strict=True)),
        "reference": dict(settings.drift_reference),
    }