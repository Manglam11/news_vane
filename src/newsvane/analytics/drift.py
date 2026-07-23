"""Has the live news drifted away from what the model was trained on?

topic_mix_shift asks "is TODAY unusual versus recent days?". Drift asks the
production question: "has the whole recent stream moved away from the training
distribution, so the model is quietly going stale?". Same maths, different
yardstick -- there I compared today to its own recent norm; here I compare the
recent window to the FROZEN training mix (settings.drift_reference).

The mix I measure is the MODEL'S OWN ANSWERS, not the sections I scraped from.
That distinction is the whole point. My scraper caps every section at the same
number of rows, so a mix built from section labels is mostly a picture of my own
quota -- if all four sections fill up, the distance is zero by arithmetic and the
model never entered the calculation at all. Grouping by predicted_label instead
puts the model back inside the statistic that claims to watch it.

The section still has a job here: it is free ground truth. The model never sees
it, so every harvest doubles as a blind exam, and agreement is the mark.
"""

from datetime import datetime

from config.settings import settings

from newsvane.analytics.distributions import TOPICS, js_divergence
from newsvane.storage.repository import agreement_counts, predicted_mix_counts


def topic_mix_drift(start: datetime, end: datetime) -> dict | None:
    """Measure how far the window's PREDICTED topic-mix has drifted from training.

    Returns {distance, threshold, is_drifting, live, reference, agreement}.
    distance is the JS divergence in [0, 1] between the model's live output mix
    and the frozen training mix; is_drifting is that distance crossing
    settings.drift_threshold.

    Returns None when the window holds no CLASSIFIED articles. Rows scraped before
    the model was wired into the harvest carry no prediction and never will, so a
    freshly-migrated table reads None until a day's worth of new rows lands. That
    silence is honest: there is no model output to measure yet.
    """
    counts = predicted_mix_counts(start, end)
    total = sum(counts.values())
    if total == 0:
        return None

    live = [counts.get(topic, 0) / total for topic in TOPICS]
    reference = [settings.drift_reference[topic] for topic in TOPICS]
    distance = js_divergence(live, reference)

    agreed, scored = agreement_counts(start, end)

    return {
        "distance": distance,
        "threshold": settings.drift_threshold,
        "is_drifting": distance > settings.drift_threshold,
        "live": dict(zip(TOPICS, live, strict=True)),
        "reference": dict(settings.drift_reference),
        # Plain accuracy against the section, NOT the macro F1 the model was
        # accepted on. Two different measures on two different populations -- I
        # keep the name honest so nobody reads this as the training score slipping.
        "agreement": {
            "agreed": agreed,
            "scored": scored,
            "rate": agreed / scored if scored else None,
        },
    }
