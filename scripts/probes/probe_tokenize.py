"""Prove the tokenizer works BEFORE the GPU ever wakes.

Three checks, all read-only: does the test split have the same row count the
baseline scored on, does a decoded row round-trip back to real text, and does
one row carry the input_ids + attention_mask DistilBERT expects. If any of these
looks wrong, I fix it here for free -- not after a wasted training run.
"""

from newsvane.models.distilbert_data import LABELS, build_split, load_tokenizer

tokenizer = load_tokenizer()

train = build_split("train", tokenizer)
test = build_split("test", tokenizer)

print(f"train rows: {len(train)}")
print(f"test rows : {len(test)}")   # must match the baseline's test count -- the fair fight
print(f"columns   : {train.column_names}")

sample = test[0]
print(f"\nlabel id  : {sample['label']}  ->  {LABELS[sample['label']]}")
print(f"n tokens  : {len(sample['input_ids'])}")
print(f"input_ids : {sample['input_ids'][:12]} ...")
print(f"attn mask : {sample['attention_mask'][:12]} ...")
print(f"\ndecoded   : {tokenizer.decode(sample['input_ids'])}")