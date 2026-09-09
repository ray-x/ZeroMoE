#!/usr/bin/env python3
import gguf
import numpy as np
from pathlib import Path

p = Path('/Users/rayxu/.lmstudio/models/Qwen/Qwen3.6-35B-A3B-UD/Qwen3.6-35B-A3B-UD-IQ2_M.gguf')
reader = gguf.GGUFReader(p)

for l in [0, 33, 34, 39]:
    t = next((x for x in reader.tensors if x.name == f'blk.{l}.ssm_a'), None)
    if t:
        data = t.data[:]
        print(f'blk.{l}.ssm_a: min={data.min():.4f} max={data.max():.4f} mean={data.mean():.4f}')
        print(f'  first 10: {data[:10]}')
