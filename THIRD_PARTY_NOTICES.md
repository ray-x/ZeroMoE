# Third-party notices

ZeroMoE is distributed under the GNU General Public License, version 3 only
(see [`LICENSE`](LICENSE)). It builds on, links against, or contains ports of
the following MIT-licensed components. Those components remain under their own
licenses; the notices below are reproduced as required.

| Component | Version / revision | How it is used | Copyright |
|---|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | `8baabd53b77d1c48c6ef8d3b620436292f3f6231` (ggml 0.27.0) | ggml is fetched at configure time and linked into every CUDA build; ZeroMoE applies its own patches. No llama.cpp runtime code is linked. | Copyright (c) 2023-2026 The ggml authors |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | Optional checkout selected by `ZEROMOE_EXLLAMAV3_SRC` | EXL3 kernels are compiled unmodified when the optional backend is enabled. | Copyright (c) 2025 Turboderp |
| [nlohmann/json](https://github.com/nlohmann/json) | v3.11.3 | Fetched by the build and linked into the API server. | Copyright (c) 2013-2022 Niels Lohmann |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | `5bcfea7b90575c80a3e54a6967b11b39afa60d5c` | Fetched by the build and linked into the API server. | Copyright (c) 2017 Yuji Hirose |

## Ported ggml quantization code

The following files contain dequantization or matmul code ported from ggml's
`ggml/src/ggml-quants.c`, `ggml/src/ggml-common.h`, and `ggml/src/ggml-impl.h`:

- `kernels/quant/iq1_dequant.cu` (IQ1_S, IQ1_M)
- `kernels/quant/iq2_matmul.cu` (IQ2_XXS, IQ2_XS, IQ2_S)
- `kernels/quant/iq3_matmul.cu` (IQ3_XXS, IQ3_S)
- `kernels/quant/iq4xs_matmul.cu` (IQ4_XS, IQ4_NL)
- `kernels/quant/kquant_matmul.cu` (Q4_K and related K-quants)
- `kernels/quant/mxfp4_matmul.cu` (MXFP4)
- `kernels/quant/q5_1_matmul.cu` (Q5_1)
- `kernels/quant/dequant_bf16.cu` (Q6_K)
- `backends/cpu/ggml_grids.cpp` (IQ grid tables)

The block layouts and reference tables originate in ggml and are used under
the MIT license above. The `third_party/exl3_torch_stub/` headers are
ZeroMoE's own minimal ATen/c10 stand-ins, not code from PyTorch or ExLlamaV3.

## MIT license text

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model weights

ZeroMoE does not ship model weights, tokenizer data, or benchmark GGUFs.
Models you download remain under their own licenses; check the model card
before redistributing a converted or repacked file.
