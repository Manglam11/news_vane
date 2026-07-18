"""Export the fine-tuned DistilBERT to ONNX, then prove the graph still agrees with it.

Torch is a 2.5 GB dependency and the free tier gives me 512 MB, so the container
cannot serve on torch. ONNX is a portable description of the same computation that
onnxruntime executes in a fraction of the space. An export is a RECORDING of one
forward pass, so I verify the recording against the original before trusting it --
a graph that loads is not the same thing as a graph that is correct.

I export through the legacy TorchScript path (dynamo=False). Torch 2.11 defaults to
the newer Dynamo exporter, and its graph runs correctly but carries shape annotations
that onnxruntime's shape inference rejects -- it read a hidden dimension of 768 where
the graph implied 4 classes, and quantization refused to run. The old path is what the
ONNX tooling was built against. I chose it after watching the new one fail downstream,
and it is one line to switch back when the exporter catches up.
"""

import numpy as np
import onnxruntime as ort
import torch
from config.settings import settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Sentences of deliberately different lengths. If the export baked in a fixed shape,
# the short one and the long one cannot both pass -- that is the point of using two.
PROBE_TEXTS = [
    "Markets fell sharply today.",
    "Researchers unveiled a new chip architecture that promises far better power "
    "efficiency for on-device machine learning workloads across mobile hardware.",
]


def main() -> None:
    model_dir = settings.distilbert_output_dir
    if not model_dir.exists():
        raise SystemExit(f"No fine-tuned model at {model_dir}. Run the training first.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Eval mode matters: dropout is random during training and must be off for a
    # recording, or the traced graph captures noise I never want at serve time.
    model.eval()

    # The fake input the trace runs on. Its VALUES are irrelevant -- only its shape
    # and dtype are, because the recorder cares about operations, not content.
    dummy = tokenizer(
        PROBE_TEXTS[0],
        return_tensors="pt",
        truncation=True,
        max_length=settings.distilbert_max_length,
        return_token_type_ids=False,
    )

    # Without this every dimension is frozen at whatever the dummy happened to be, and
    # the graph would reject any batch size or sentence length but that one. I trained
    # with per-batch padding, so sequence length genuinely varies at serve time.
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.distilbert_onnx_path

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        # Opset 17 is the well-trodden target for transformer exports -- new enough to
        # carry the ops DistilBERT needs, old enough that every downstream tool groks it.
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"exported -> {out_path}")
    print(f"size      : {out_path.stat().st_size / 1024**2:.1f} MB")

    # A file on disk proves nothing about correctness. I run both models on the same
    # inputs and compare the raw logits -- if the recording missed something, the
    # numbers diverge here rather than silently in production.
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    for text in PROBE_TEXTS:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.distilbert_max_length,
            return_token_type_ids=False,
        )

        with torch.no_grad():
            torch_logits = model(**enc).logits.numpy()

        onnx_logits = session.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy(),
            },
        )[0]

        gap = float(np.abs(torch_logits - onnx_logits).max())
        same = int(torch_logits.argmax()) == int(onnx_logits.argmax())
        print(f"seq_len={enc['input_ids'].shape[1]:>3}  max_diff={gap:.6f}  same_label={same}")


if __name__ == "__main__":
    main()
