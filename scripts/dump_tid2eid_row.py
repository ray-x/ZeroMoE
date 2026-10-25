import os
import sys

llama_src = os.environ.get("LLAMA_SRC")
if not llama_src:
    raise SystemExit("set LLAMA_SRC to the llama.cpp source tree")
sys.path.insert(0, os.path.join(llama_src, "gguf-py"))
from gguf.gguf_reader import GGUFReader
import numpy as np

path = sys.argv[1]

path = sys.argv[1]
token_ids = [int(x) for x in sys.argv[2:]] or [43399, 117844, 50845, 63104, 69146]

r = GGUFReader(path)
t = None
for tensor in r.tensors:
    if tensor.name == "blk.0.ffn_gate_tid2eid.weight":
        t = tensor
        break

if t is None:
    print("tensor not found in this file")
    sys.exit(1)

print(f"raw tensor.shape (ne[], file order): {t.shape}")
print(f"raw t.data.shape as returned by GGUFReader: {t.data.shape}, dtype: {t.data.dtype}")

flat = np.asarray(t.data).reshape(-1)
topk = int(t.shape[0])   # ne[0]
n_vocab = int(t.shape[1])  # ne[1]
print(f"reinterpreting as n_vocab={n_vocab}, topk={topk}, total elements={flat.size} "
      f"(expected {n_vocab*topk})")

arr = flat.reshape(n_vocab, topk)

for tok in token_ids:
    if tok < n_vocab:
        print(f"token_id={tok}: row={arr[tok].tolist()}")
    else:
        print(f"token_id={tok}: OUT OF BOUNDS (n_vocab={n_vocab})")
