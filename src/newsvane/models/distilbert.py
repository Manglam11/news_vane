"""The fine-tuned DistilBERT topic classifier, served behind the MODEL contract.

This replaces the baseline. The contract is untouched -- predict(text) still
returns {label, score, sentiment} -- so the API, storage and every test stay
exactly as they were. Only the brain changed, and it entered through the same
door the dummy and the baseline used. That is the Lego promise, cashed in a
second time.

The model earned this swap on evidence, not hype: macro F1 0.9452 vs the
baseline's 0.9060, and it cut the Business <-> Sci/Tech confusions from 315
to 223 -- the exact weakness it existed to close.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from config.settings import settings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@lru_cache(maxsize=1)
def load_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the fine-tuned model + its tokenizer once and reuse them forever.

    Same reasoning as the baseline: reading weights off disk is slow, so the
    cache turns it into a one-time cost paid on the first request. I load onto
    the CPU on purpose -- serving one short headline is trivial work, and a CPU
    model needs no GPU on the box that hosts the API. If the folder is missing I
    fail loudly rather than let the API serve nonsense.

    The label order is not re-derived and hoped to match -- it is read straight
    from the model's own config (id2label), baked in at training time.
    """
    path = settings.distilbert_output_dir
    if not path.exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {path}. "
            f"Run `uv run python -m scripts.train_distilbert` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict(text: str) -> dict:
    """Classify one article and report how confident the model actually is.

    A softmax over the 4 logits turns raw scores into probabilities that sum to
    1, exactly like the baseline's predict_proba -- so 'score' means the same
    thing to the anomaly and drift maths downstream. I take the winner as the
    label and its probability as the score.

    torch.no_grad() switches off gradient bookkeeping: I am serving, not
    training, so there is no backward pass to prepare for and it is wasted work.

    Sentiment stays an honest 'neutral' placeholder -- this is still a topic
    classifier. It gets filled in when the sentiment model arrives, without
    touching this contract.
    """
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=settings.distilbert_max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probabilities = F.softmax(logits, dim=-1)
    best = int(probabilities.argmax())

    return {
        "label": model.config.id2label[best],
        "score": float(probabilities[best]),
        "sentiment": "neutral",
    }