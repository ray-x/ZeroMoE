import os
import sys

llama_src = os.environ.get("LLAMA_SRC")
if not llama_src:
    raise SystemExit("set LLAMA_SRC to the llama.cpp source tree")
sys.path.insert(0, os.path.join(llama_src, "gguf-py"))
from gguf.gguf_reader import GGUFReader
import numpy as np

path = sys.argv[1]

r = GGUFReader(path)
for want in ["blk.0.ffn_norm.weight", "blk.0.attn_norm.weight"]:
    t = None
    for tensor in r.tensors:
        if tensor.name == want:
            t = tensor
            break
    if t is None:
        print(f"{want}: not found in this file")
        continue
    arr = np.asarray(t.data).reshape(-1).astype(np.float32)
    print(f"{want}: shape={t.shape}, dtype={t.data.dtype}, n={arr.size}")
    print(f"  min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} "
          f"std={arr.std():.4f}")
    print(f"  first 10 values: {arr[:10].tolist()}")
