"""Re-prove macro F1 on the ONNX graphs, on the same frozen test set as the bar.

A smaller model is only an upgrade if it is no worse at its job. AG News ships a
fixed 7,600-row test set, so all three numbers -- baseline 0.9060, DistilBERT
0.9452, and int8 -- sit on one honest scale.

I evaluate the float32 graph FIRST as a control. Trainer used to batch and pad for
me; here I do it by hand, so a mistake in my loop would look exactly like a loss
from quantization. The float32 graph already matched PyTorch to six decimals, so
it must reproduce 0.9452 -- if it does, the harness is proven and the int8 number
can be believed. Measure the instrument before measuring with it.
"""

import mlflow
import numpy as np
import onnxruntime as ort
from config.settings import settings
from sklearn.metrics import classification_report, f1_score
from transformers import DataCollatorWithPadding

from newsvane.models.distilbert_data import LABELS, build_split, load_tokenizer
from scripts.train import setup_tracking
from scripts.train_distilbert import log_evaluation

TORCH_BAR = 0.9452
BASELINE_BAR = 0.9060


def run_graph(path, dataset, collator, batch_size: int) -> np.ndarray:
    """Every row through one ONNX graph, batched and padded by hand."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    predictions = []

    for start in range(0, len(dataset), batch_size):
        chunk = dataset[start : start + batch_size]
        rows = [
            {
                "input_ids": chunk["input_ids"][i],
                "attention_mask": chunk["attention_mask"][i],
            }
            for i in range(len(chunk["label"]))
        ]
        # Pad this batch to its own longest row -- the same rule the training
        # collator used. ONNX needs a rectangle; the dataset stores ragged rows.
        batch = collator(rows)
        logits = session.run(
            ["logits"],
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )[0]
        predictions.append(np.argmax(logits, axis=-1))

    return np.concatenate(predictions)


def score(y_true, y_pred) -> tuple[float, float, dict]:
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    ids = list(range(len(LABELS)))
    per_class = f1_score(y_true, y_pred, average=None, labels=ids)
    named = {
        f"f1_{LABELS[i].replace('/', '_').lower()}": value
        for i, value in enumerate(per_class)
    }
    return macro, weighted, named


def main() -> None:
    setup_tracking()

    for path in (settings.distilbert_onnx_path, settings.distilbert_onnx_int8_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the export and quantize first.")

    tokenizer = load_tokenizer()
    dataset = build_split("test", tokenizer)
    y_true = np.array(dataset["label"])
    # numpy tensors, so this evaluation never touches torch at all -- a preview of
    # the serving container, which will not have torch installed.
    collator = DataCollatorWithPadding(tokenizer, return_tensors="np")
    batch_size = settings.distilbert_eval_batch_size

    print(f"test rows: {len(dataset)}")

    print("\nfloat32 graph (harness control)...")
    fp32_pred = run_graph(
        settings.distilbert_onnx_path, dataset, collator, batch_size
    )
    fp32_macro, fp32_weighted, _ = score(y_true, fp32_pred)
    print(f"  macro F1 : {fp32_macro:.4f}   (torch was {TORCH_BAR:.4f})")

    print("\nint8 graph...")
    int8_pred = run_graph(
        settings.distilbert_onnx_int8_path, dataset, collator, batch_size
    )
    int8_macro, int8_weighted, int8_per_class = score(y_true, int8_pred)

    size_mb = settings.distilbert_onnx_int8_path.stat().st_size / 1024**2

    with mlflow.start_run(run_name="distilbert-onnx-int8"):
        mlflow.log_params(
            {
                "model": "distilbert onnx int8 dynamic",
                "opset": 17,
                "max_length": settings.distilbert_max_length,
                "test_rows": len(dataset),
            }
        )
        mlflow.log_metrics(
            {
                "macro_f1": int8_macro,
                "weighted_f1": int8_weighted,
                # The control, logged beside the result so the run records not just
                # the number but the evidence that the number was measured correctly.
                "control_fp32_macro_f1": fp32_macro,
                "model_size_mb": size_mb,
                **int8_per_class,
            }
        )
        log_evaluation(y_true, int8_pred)
        run_id = mlflow.active_run().info.run_id

    drop = TORCH_BAR - int8_macro
    print("\n--- verdict ---")
    print(f"baseline (TF-IDF + NB) : {BASELINE_BAR:.4f}")
    print(f"torch DistilBERT       : {TORCH_BAR:.4f}   255.4 MB")
    print(f"onnx float32 (control) : {fp32_macro:.4f}")
    print(f"onnx int8              : {int8_macro:.4f}   {size_mb:.1f} MB")
    print(f"cost of quantization   : {drop:+.4f} macro F1")
    print(f"\n{classification_report(y_true, int8_pred, target_names=LABELS, digits=4)}")
    print(f"mlflow run -> {run_id}")


if __name__ == "__main__":
    main()