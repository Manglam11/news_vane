"""The MODEL box: predict(text) -> {label, score, sentiment}.

Everything downstream imports predict from here, never from a specific model
module. That indirection is what lets me change the brain without touching a
single caller -- the dummy came out, the baseline went in, and now DistilBERT
arrives, each by editing this one import line and nothing else.
"""

from newsvane.models.distilbert import predict

__all__ = ["predict"]