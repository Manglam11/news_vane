"""The ANALYTICS box's one door -- everything downstream calls this, nothing else.

My blueprint froze this contract: summarise(range) -> {trends, distribution,
anomalies, drift}. The three maths engines each live in their own file, but the
API and the dashboard should never need to know that. They ask ONE question --
"what is the news doing between these two times?" -- and get one answer back.

The fourth gauge is lit now. drift compares the live window against the model's
frozen TRAINING mix -- the Phase 6 job the contract always held a slot for. The
door never changed shape; the value that was None simply filled in.
"""

from datetime import datetime

from newsvane.analytics.anomalies import volume_anomalies
from newsvane.analytics.distributions import topic_mix_shift
from newsvane.analytics.drift import topic_mix_drift
from newsvane.analytics.trends import topic_momentum


def summarise(start: datetime, end: datetime) -> dict:
    """Read the stored articles across a window and report the pulse of the news.

    One call, four answers -- the frozen ANALYTICS contract:
      trends       -- each topic's article-count per day (momentum)
      distribution -- today's topic-mix vs the recent norm (shape shift)
      anomalies    -- topics whose volume spiked far past their own normal
      drift        -- live-vs-training distance (fires when the ground moves)
    """
    return {
        "trends": topic_momentum(start, end),
        "distribution": topic_mix_shift(start, end),
        "anomalies": volume_anomalies(start, end),
        "drift": topic_mix_drift(start, end),
    }