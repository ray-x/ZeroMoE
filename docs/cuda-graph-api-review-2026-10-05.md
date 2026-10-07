# CUDA graph API review for MoEStream

This review checks the proposed stream-priority, capture, graph-memory, TMA,
and host-event ideas against the engine's current GGML graph boundary and
NVIDIA's CUDA 13.4 documentation. It is design guidance; it does not enable
any new execution path.

| Proposal | Assessment for this engine |
| --- | --- |
| High-priority compute and low-priority prefetch streams | A priority is only a scheduling hint for kernels. CUDA documents no effect on H2D/D2H operations and no preemption of running kernels. The graph compute stream is owned by the GGML CUDA backend, while `GpuTopology`'s `compute_stream` and copy streams serve other paths. Changing those topology streams would not prioritize the GGML graph. Do not expect this to arbitrate PCIe/NVMe transfer bandwidth. |
| Capture the whole GGML scheduler externally, then inject conditionals | GGML already captures its CUDA backend work. The scheduler can contain several backend graphs and CPU work, so an outer capture around the scheduler call would not capture arbitrary host execution and may collide with GGML's own capture. Keep using the captured pure-CUDA child graphs and a parent graph for device predicates/conditionals. Conditional bodies permit kernels, child graphs, memcpy, memset, empty and nested conditional nodes; host nodes are prohibited there. |
| Allocate variable scratch inside conditional graph bodies | Graph allocation nodes have graph-defined addresses and GPU-ordered lifetimes; CUDA may reuse physical storage when those lifetimes do not overlap. They do not size allocations from a runtime expert count. Allocation/free nodes are also outside the allowed conditional-body node set. They may help with fixed-lifetime top-level scratch later, but do not replace the bounded, pre-sized staging pools for the current branch design. |
| Use TMA for the RTX 5060 Ti | The 5060 Ti is compute capability 12.0; the RTX 3090 is 8.6. TMA is available on supported newer architectures and can move tiles between global memory and shared memory inside a custom kernel. It does not move SSD/host data to VRAM, arbitrate PCIe, or automatically accelerate GGML/cuBLAS kernels. A future attention-kernel port would need an SM120-specific implementation and an A/B on the 5060 Ti. |
| Treat `cudaEventRecordWithFlags(..., cudaEventRecordExternal)` as a nonblocking host callback | These are different mechanisms. External event recording makes an event record visible when capturing a stream graph. It does not run host work or make a host function nonblocking. `cudaLaunchHostFunc` blocks dependent stream progress until the callback returns and the callback cannot call CUDA APIs. For CPU work to overlap independent GPU work, use an external host worker that waits for a GPU event and publish a completion dependency before the merge. The early-ready scheduler already follows that shape. |

Sources: [CUDA stream management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html), [conditional-node body requirements](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaConditionalNodeParams.html), [CUDA graph memory nodes and conditional graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [host callback rules](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html), and [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus).

The current implementation aligns with these constraints by keeping host
callbacks out of conditional bodies, using GGML's captured CUDA graphs as
children where they are single-backend, and retaining explicit stable staging
storage. The asynchronous SSD route remains host-mediated; GDS/cuFile is not
used on this box.
