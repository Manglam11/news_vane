"""A quick, throwaway check: can PyTorch actually see my RTX 4060?

The whole point of installing the +cu128 build was GPU training. But an import
that succeeds proves nothing -- torch imports fine even when it silently falls
back to the CPU. The only honest proof is asking CUDA directly and reading back
my card's name. If this prints False, B3 would train on the CPU for hours while
the 4060 sits idle, so I gate the whole phase on this one answer.
"""

import torch

print(f"torch version   : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    print(f"GPU count       : {torch.cuda.device_count()}")
else:
    print("!! CUDA NOT visible -- torch would train on CPU. Stop and fix before B3.")
