"""The MODEL box: predict(text) -> {label, score, sentiment}.

Everything downstream imports predict from here, never from a specific model
module. That indirection is what lets me change the brain without touching a
single caller -- the dummy came out and the trained baseline went in by editing
this one import line. DistilBERT will arrive the same way in Phase 5.
"""

from newsvane.models.baseline import predict

__all__ = ["predict"]
