"""The MODEL box: predict(text) -> {label, score, sentiment}.

Everything downstream imports predict from here, never from a specific model
module. That indirection is what lets me change the brain without touching a
single caller -- the dummy came out, the baseline went in, DistilBERT replaced
it, and the quantized ONNX graph took over, each by editing one import line.

There are now TWO engines behind this one door: a topic classifier and a mood
reader. This module is where they become the single three-key answer my
blueprint froze. Neither engine knows the other exists, and no caller knows
there are two -- which is the whole point of a box having one door.
"""

from newsvane.models.onnx import predict as classify_topic
from newsvane.models.sentiment import read_sentiment

__all__ = ["predict"]


def predict(text: str) -> dict:
    """Read one article's topic and its mood, in the shape the contract promises.

    The topic brain still hands back its own "neutral" placeholder and I
    deliberately overwrite it rather than delete it. That module's contract is
    its own business: it promised to stay a topic classifier and to let the
    sentiment fill in from outside, and it did exactly that.

    read_sentiment also returns the raw compound score, which I drop here. The
    contract carries a word a human can read; the number ANALYTICS averages is
    fetched by the one caller inside this box that actually needs it.
    """
    mood, _compound = read_sentiment(text)
    return {**classify_topic(text), "sentiment": mood}
