"""Turn AG News text into the numbers DistilBERT reads.

I pull every row through the DATA box's one door -- the same door the baseline
used -- so DistilBERT is trained and tested on byte-identical text. A second
reader of the CSV would risk a quiet mismatch, and the fair fight would be lost
before it even started.
"""

from config.settings import settings
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from newsvane.data import get_articles

# Alphabetical on purpose: this matches the baseline's sorted() label order, so the
# two confusion matrices line up class-for-class when I compare them in B4.
LABELS = ["Business", "Sci/Tech", "Sports", "World"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}


def load_tokenizer() -> PreTrainedTokenizerBase:
    """The checkpoint's own tokenizer -- its vocabulary must match its weights."""
    return AutoTokenizer.from_pretrained(settings.distilbert_checkpoint)


def build_split(split: str, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """One split, pulled through the DATA box and tokenized into model-ready rows."""
    articles = get_articles("kaggle", split)
    dataset = Dataset.from_dict(
        {
            "text": [article["text"] for article in articles],
            "label": [LABEL_TO_ID[article["topic"]] for article in articles],
        }
    )

    def tokenize(batch: dict) -> dict:
        # Truncate long rows to the lunchbox size. I do NOT pad here -- the training
        # collator pads each batch to its own longest row instead (B3). And I drop
        # token_type_ids: DistilBERT has no segment embeddings, so it never reads them.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=settings.distilbert_max_length,
            return_token_type_ids=False,
        )

    return dataset.map(tokenize, batched=True)
