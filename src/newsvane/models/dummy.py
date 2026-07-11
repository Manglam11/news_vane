"""Placeholder prediction logic for the walking skeleton.

Right now this box does no real ML. It exists only to prove the MODEL
contract -- predict(text) -> {label, score, sentiment} -- so I can wire
and test the rest of the pipeline (API, storage, dashboard) end to end.
I'll swap these insides for the TF-IDF + Naive Bayes baseline in Phase 1
without touching this function's shape.
"""


def predict(text: str) -> dict:
    """Return a fixed, honest 'nothing learned yet' prediction.

    The values are deliberately constant: label 'unknown', zero
    confidence, neutral sentiment. That keeps it obvious this is a
    stand-in, not a model that has actually been trained on anything.
    """
    return {
        "label": "unknown",
        "score": 0.0,
        "sentiment": "neutral",
    }
