"""A throwaway training run on 200 rows -- proof the machinery turns BEFORE the
real thing. I check three things: the model lands on the GPU, one training step
completes without error, and a prediction comes back the right shape. Nothing
here is saved or logged; it exists only to fail fast and cheap.
"""

import torch
from config.settings import settings
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from newsvane.models.distilbert_data import LABELS, build_split, load_tokenizer

tokenizer = load_tokenizer()

# Tiny slices -- just enough to run a handful of steps.
train_ds = build_split("train", tokenizer).select(range(200))
test_ds = build_split("test", tokenizer).select(range(100))

model = AutoModelForSequenceClassification.from_pretrained(
    settings.distilbert_checkpoint,
    num_labels=len(LABELS),
)

args = TrainingArguments(
    output_dir="models/_smoke",  # throwaway
    max_steps=10,  # stop after 10 steps, not 3 epochs
    per_device_train_batch_size=16,
    fp16=settings.distilbert_fp16,
    report_to="none",
    logging_steps=1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

print(f"CUDA available : {torch.cuda.is_available()}")
print(f"model device   : {next(model.parameters()).device}")

trainer.train()

preds = trainer.predict(test_ds)
print(f"\nprediction shape : {preds.predictions.shape}   (want: (100, 4))")
print(f"model device now : {next(model.parameters()).device}   (want: cuda:0)")
print("\nSMOKE OK -- loop turns, GPU used, shape correct.")
