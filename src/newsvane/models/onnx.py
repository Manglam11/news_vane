"""The int8 ONNX topic classifier -- the brain that actually ships.

Same weights, same contract, no torch. The fine-tuned PyTorch model was a
TRAINING artefact: 255 MB of float32 plus a ~2.5 GB runtime, which never had a
hope of fitting inside a 512 MB free tier. This module serves the quantized
graph instead -- 64.2 MB, macro F1 0.9462 against torch's 0.9452, proven on the
same frozen 7,600-row test set.

What transformers used to do for me, I now do by hand: tokenize, feed two int64
arrays, softmax the logits in numpy. Three small pieces of arithmetic in
exchange for dropping torch and transformers from the serving image.
"""

import json
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from config.settings import settings
from tokenizers import Tokenizer


@lru_cache(maxsize=1)
def load_session() -> tuple[Tokenizer, ort.InferenceSession, dict[int, str]]:
    """Load the graph, its tokenizer and its label map once, then reuse forever.

    Same reasoning as every brain before it: reading 64 MB off disk is slow, so
    the cache turns it into a one-time cost paid on the first request. CPU only
    -- classifying one headline is trivial work and the API host has no GPU.

    Truncation has to be switched on explicitly here. transformers used to read
    max_length out of tokenizer_config.json for me; a raw Tokenizer does not, so
    a long article would happily produce thousands of tokens and feed a graph
    that was trained on 128. I fail loudly if the artefacts are missing rather
    than let the API serve nonsense.
    """
    graph_path = settings.distilbert_onnx_int8_path
    if not graph_path.exists():
        raise FileNotFoundError(
            f"No quantized graph at {graph_path}. Run "
            "`uv run python -m scripts.export_onnx` then "
            "`uv run python -m scripts.quantize_onnx`."
        )

    tokenizer = Tokenizer.from_file(str(settings.distilbert_tokenizer_path))
    tokenizer.enable_truncation(max_length=settings.distilbert_max_length)

    session = ort.InferenceSession(
        str(graph_path),
        providers=["CPUExecutionProvider"],
    )

    # The label order is read from the model's own config, baked in at training
    # time -- never re-derived and hoped to match. JSON keys are always strings,
    # so I cast them back to the ints that argmax will hand me.
    config = json.loads(settings.distilbert_config_path.read_text(encoding="utf-8"))
    id2label = {int(index): label for index, label in config["id2label"].items()}

    return tokenizer, session, id2label


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw logits into probabilities that sum to 1.

    I subtract the max first. The maths is identical either way, but exp() of a
    large logit overflows to infinity in float32 and the whole vector becomes
    NaN -- shifting everything down so the largest value is 0 keeps every
    exponent safely below 1 and costs nothing.
    """
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def predict(text: str) -> dict:
    """Classify one article and report how confident the model actually is.

    The tokenizer adds [CLS] and [SEP] itself via its own post-processor, so I
    pass the text through untouched. Both inputs go in as int64 with a leading
    batch dimension of 1 -- the graph declares batch and sequence as dynamic
    axes, so one short headline is a perfectly valid batch of one.

    'score' means exactly what it meant under the baseline and under torch: the
    winning class's probability. The anomaly and drift maths downstream read
    this number, so its meaning is not mine to change.

    Sentiment stays an honest 'neutral' placeholder -- this is still a topic
    classifier. It gets filled in when the sentiment model arrives, without
    touching this contract.
    """
    tokenizer, session, id2label = load_session()

    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    logits = session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0][0]

    probabilities = softmax(logits)
    best = int(probabilities.argmax())

    return {
        "label": id2label[best],
        "score": float(probabilities[best]),
        "sentiment": "neutral",
    }
