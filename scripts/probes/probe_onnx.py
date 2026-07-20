"""A look at what the int8 graph actually expects, before I write the serving module.

I am about to hand-roll the tokenize -> feed -> softmax path that transformers
used to do for me. Guessing the input names or assuming the tokenizer adds its
own special tokens would mean writing code against a model I have not read.
"""

import json

import onnxruntime as ort
from config.settings import settings
from tokenizers import Tokenizer

session = ort.InferenceSession(
    str(settings.distilbert_onnx_int8_path),
    providers=["CPUExecutionProvider"],
)

print("inputs:")
for i in session.get_inputs():
    print(f"  {i.name}  {i.type}  {i.shape}")

print("outputs:")
for o in session.get_outputs():
    print(f"  {o.name}  {o.type}  {o.shape}")

config = json.loads((settings.distilbert_output_dir / "config.json").read_text())
print("id2label:", config["id2label"])

tokenizer = Tokenizer.from_file(str(settings.distilbert_output_dir / "tokenizer.json"))
encoding = tokenizer.encode("Nvidia unveils a new AI chip for data centres")
print("tokens:", encoding.tokens)
print("ids:", encoding.ids)
print("attention_mask:", encoding.attention_mask)
