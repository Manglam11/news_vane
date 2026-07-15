"""The ANALYTICS box's one door -- everything downstream calls this, nothing else.

My blueprint froze this contract: summarise(range) -> {trends, distribution,
anomalies, drift}. The three maths engines each live in their own file, but the
API and the dashboard should never need to know that. They ask ONE question --
"what is the news doing between these two times?" -- and get one answer back.

drift is deliberately None for now. It compares live news against the model's
TRAINING distribution, which is Phase 6 work. But the contract names four keys,
and a contract that changes shape later was never frozen. So I honour the shape
today and let the fourth gauge light up in Phase 6 -- the door never changes.
"""

from datetime import datetime

from newsvane.analytics.anomalies import volume_anomalies
from newsvane.analytics.distributions import topic_mix_shift
from newsvane.analytics.trends import topic_momentum


def summarise(start: datetime, end: datetime) -> dict:
    """Read the stored articles across a window and report the pulse of the news.

    One call, four answers -- the frozen ANALYTICS contract:
      trends       -- each topic's article-count per day (momentum)
      distribution -- today's topic-mix vs the recent norm (shape shift)
      anomalies    -- topics whose volume spiked far past their own normal
      drift        -- live-vs-training distance (Phase 6; None until then)
    """
    return {
        "trends": topic_momentum(start, end),
        "distribution": topic_mix_shift(start, end),
        "anomalies": volume_anomalies(start, end),
        "drift": None,
    }