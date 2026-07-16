"""Fine-tune DistilBERT on AG News and log the run beside the baseline.

Same DATA door, same test set, same MLflow experiment as the baseline -- so the
two runs sit side by side in the UI and the comparison in B4 is apples to apples.
Every knob comes from settings; a retrain is a config edit, never a code edit.
"""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
from config.settings import settings
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from newsvane.models.distilbert_data import LABELS, build_split, load_tokenizer
from scripts.train import setup_tracking

ID_TO_LABEL = dict(enumerate(LABELS))
LABEL_TO_ID = {name: index for index, name in ID_TO_LABEL.items()}


def compute_metrics(eval_pred) -> dict:
    """Macro F1 each epoch, so I can watch it climb instead of waiting blind."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def log_evaluation(y_true, y_pred) -> None:
    """The confusion matrix + per-class report, as a text artefact -- same shape the
    baseline logged, so I can lay the two grids next to each other in B4."""
    ids = list(range(len(LABELS)))
    lines = [
        "confusion matrix (rows = true, cols = predicted)",
        f"labels: {LABELS}",
        str(confusion_matrix(y_true, y_pred, labels=ids)),
        "",
        "per-class report",
        classification_report(y_true, y_pred, target_names=LABELS, digits=4),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        mlflow.log_artifact(str(path))


def main() -> None:
    setup_tracking()

    tokenizer = load_tokenizer()
    train_ds = build_split("train", tokenizer)
    test_ds = build_split("test", tokenizer)
    print(f"train: {len(train_ds)} rows | test: {len(test_ds)} rows")

    # DistilBERT arrives knowing language but not our 4 classes. from_pretrained bolts
    # a fresh 4-way head on top; id2label bakes the class order into the saved config,
    # so the loaded model is self-describing and never re-derives its labels.
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.distilbert_checkpoint,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    args = TrainingArguments(
        output_dir=settings.distilbert_output_dir,
        num_train_epochs=settings.distilbert_epochs,
        per_device_train_batch_size=settings.distilbert_train_batch_size,
        per_device_eval_batch_size=settings.distilbert_eval_batch_size,
        learning_rate=settings.distilbert_learning_rate,
        weight_decay=settings.distilbert_weight_decay,
        warmup_ratio=settings.distilbert_warmup_ratio,
        eval_strategy="epoch",
        logging_steps=100,
        fp16=settings.distilbert_fp16,
        report_to="none",  # I drive MLflow myself, mirroring the baseline -- no auto-logging
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="distilbert"):
        mlflow.log_params(
            {
                "model": settings.distilbert_checkpoint,
                "epochs": settings.distilbert_epochs,
                "train_batch_size": settings.distilbert_train_batch_size,
                "learning_rate": settings.distilbert_learning_rate,
                "weight_decay": settings.distilbert_weight_decay,
                "warmup_ratio": settings.distilbert_warmup_ratio,
                "max_length": settings.distilbert_max_length,
                "train_rows": len(train_ds),
                "test_rows": len(test_ds),
            }
        )

        trainer.train()

        # One honest pass over the frozen test set for the real numbers + the matrix.
        output = trainer.predict(test_ds)
        y_pred = np.argmax(output.predictions, axis=-1)
        y_true = output.label_ids

        macro = f1_score(y_true, y_pred, average="macro")
        weighted = f1_score(y_true, y_pred, average="weighted")
        per_class = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))))
        class_metrics = {
            f"f1_{LABELS[i].replace('/', '_').lower()}": score
            for i, score in enumerate(per_class)
        }

        mlflow.log_metrics({"macro_f1": macro, "weighted_f1": weighted, **class_metrics})
        log_evaluation(y_true, y_pred)

        trainer.save_model(settings.distilbert_output_dir)
        tokenizer.save_pretrained(settings.distilbert_output_dir)

        print(f"\nmacro F1    : {macro:.4f}   (baseline bar: 0.9060)")
        print(f"weighted F1 : {weighted:.4f}")
        print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))
        print(f"saved -> {settings.distilbert_output_dir}")
        print(f"mlflow run -> {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()