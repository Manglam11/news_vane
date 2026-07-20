"""The MODEL box: predict(text) -> {label, score, sentiment}.

Everything downstream imports predict from here, never from a specific model
module. That indirection is what lets me change the brain without touching a
single caller -- the dummy came out, the baseline went in, DistilBERT replaced
it, and now the quantized ONNX graph takes over, each by editing this one
import line and nothing else.
"""

from newsvane.models.onnx import predict

__all__ = ["predict"]
