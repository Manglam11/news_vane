"""Shrink the ONNX graph to int8 so it fits inside a 512 MB free tier.

Training needs float32 precision because gradient updates are tiny. Serving does
not -- one forward pass barely depends on the seventh decimal place. Dynamic
quantization stores each weight as an 8-bit integer plus a scale factor, roughly
a quarter of the size. "Dynamic" means the activation ranges are measured per
input at runtime, so I need no calibration dataset.

This script only reports SIZE. Whether the smaller model is still as ACCURATE is
a separate question, answered on the full test set -- a shrink that costs macro F1
is a downgrade wearing a costume.
"""

import numpy as np
import onnxruntime as ort
from config.settings import settings
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

PROBE_TEXTS = [
    "Markets fell sharply today.",
    "Researchers unveiled a new chip architecture that promises far better power "
    "efficiency for on-device machine learning workloads across mobile hardware.",
]


def main() -> None:
    src = settings.distilbert_onnx_path
    if not src.exists():
        raise SystemExit(f"No ONNX graph at {src}. Run the export first.")

    dst = settings.distilbert_onnx_int8_path

    quantize_dynamic(
        model_input=src,
        model_output=dst,
        # QInt8 is signed 8-bit. I keep the default rather than reaching for QUInt8:
        # the choice is a hardware-performance trade-off, and I have no evidence yet
        # that it matters here. Evidence first, tuning second.
        weight_type=QuantType.QInt8,
    )

    src_mb = src.stat().st_size / 1024**2
    dst_mb = dst.stat().st_size / 1024**2

    print(f"float32 -> {src_mb:.1f} MB")
    print(f"int8    -> {dst_mb:.1f} MB   ({src_mb / dst_mb:.1f}x smaller)")

    # A quick smell test, not the verdict. I compare the two graphs on two sentences
    # to catch a catastrophic break early; the honest measurement is the full test set.
    tokenizer = AutoTokenizer.from_pretrained(settings.distilbert_output_dir)
    fp32 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    for text in PROBE_TEXTS:
        enc = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=settings.distilbert_max_length,
            return_token_type_ids=False,
        )
        feed = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

        a = fp32.run(["logits"], feed)[0]
        b = int8.run(["logits"], feed)[0]

        gap = float(np.abs(a - b).max())
        same = int(a.argmax()) == int(b.argmax())
        print(f"seq_len={enc['input_ids'].shape[1]:>3}  max_diff={gap:.4f}  same_label={same}")


if __name__ == "__main__":
    main()
