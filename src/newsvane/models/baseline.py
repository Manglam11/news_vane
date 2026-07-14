"""The trained TF-IDF + Naive Bayes baseline, served behind the MODEL contract.

This replaces the dummy. The contract is untouched -- predict(text) still returns
{label, score, sentiment} -- so the API, the storage layer and every test stay
exactly as they were. Only the insides changed. That is the whole promise of the
Lego contract, and this is the first time it gets cashed in.

The fitted pipeline carries its own vectoriser, so I hand it raw text and it does
the TF-IDF transform itself using the vocabulary it learned at training time. No
cleaning or vectorising is re-implemented here -- if it were, training and serving
could drift apart without anything ever raising an error.
"""

from functools import lru_cache

import joblib
from config.settings import settings
from sklearn.pipeline import Pipeline


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Load the fitted pipeline once and reuse it for every request.

    Reading a model off disk is slow, and the API would otherwise do it on every
    single call. The cache turns it into a one-time cost paid on the first
    request. If the file is missing I fail loudly here rather than let the API
    serve nonsense -- a model that silently isn't there is worse than a crash.
    """
    path = settings.baseline_model_path
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `uv run python -m scripts.train` first."
        )
    return joblib.load(path)


def predict(text: str) -> dict:
    """Classify one article and report how confident the model actually is.

    Naive Bayes gives me a probability for every class, not just a winner. I take
    the winner as the label and its probability as the score, because a prediction
    without a confidence is useless downstream -- the anomaly and drift maths in
    Phase 4 needs to know when the model was unsure.

    Sentiment is still a placeholder. The baseline is a topic classifier and
    nothing more, and I would rather return an honest 'neutral' than invent a mood
    I have not built yet. It gets filled in when the sentiment model arrives.
    """
    model = load_model()

    probabilities = model.predict_proba([text])[0]
    best = probabilities.argmax()

    return {
        "label": str(model.classes_[best]),
        "score": float(probabilities[best]),
        "sentiment": "neutral",
    }
