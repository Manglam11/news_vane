"""The mood reading -- the MODEL box's second engine.

VADER is a lexicon of ~7,500 scored words plus rules for everything a raw word
count misses: negation ("not good"), intensifiers ("very good"), capitals
("GOOD") and punctuation ("good!!!"). It returns a compound score in [-1, 1],
which I cut into three labels at the thresholds in settings.

It is a BASELINE, and deliberately so. The section URL handed me a free topic
label, so the topic classifier could be fine-tuned and PROVEN -- 0.9060 to
0.9462 on a frozen test set. Nothing on a news page tells me the right MOOD, so
a sentiment transformer could only ever be assumed better, never measured
better, and my own acceptance rule refuses an upgrade I cannot evidence. The
road back is written down rather than left implied: hand-label a few thousand
of my own scraped articles, and the gate can finally run.
"""

from functools import lru_cache

from config.settings import settings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def load_analyzer() -> SentimentIntensityAnalyzer:
    """Build the analyzer once and reuse it forever.

    Same reasoning as the ONNX session next door: constructing it parses a
    434 KB lexicon off disk, and paying that per article would cost far more
    than the scoring itself. The lexicon travels inside the installed package,
    so there is no start-up download and nothing extra to ship into the image.
    """
    return SentimentIntensityAnalyzer()


def label_for(compound: float) -> str:
    """Cut the [-1, 1] compound score into the three words the contract carries.

    Both boundaries count TOWARDS the pole: exactly 0.05 is positive, exactly
    -0.05 is negative. That is VADER's own published convention and I keep it
    rather than inventing my own -- but it is a decision, not an accident, so it
    gets pinned by a test exactly like the scraper's 72-hour gate.
    """
    if compound >= settings.sentiment_positive_threshold:
        return "positive"
    if compound <= settings.sentiment_negative_threshold:
        return "negative"
    return "neutral"


def read_sentiment(text: str) -> tuple[str, float]:
    """Read one article's mood as (label, compound score).

    I hand back both halves because they answer two different questions. The
    label is what the MODEL contract carries and what a human reads on the
    dashboard. The raw compound is what ANALYTICS averages -- you cannot take
    the mean of three words, and a mood series needs a number.
    """
    compound = load_analyzer().polarity_scores(text)["compound"]
    return label_for(compound), compound
