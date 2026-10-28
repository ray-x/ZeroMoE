c++ -std=c++23 -O2 \
    -Icore -Imodels/base -Imemory -Imodels/graph -Ibackends/ggml -Ibackends/cpu \
    tools/gguf_tool.cpp tools/gguf_pack.cpp tools/gguf_io.cpp tools/gguf_verify.cpp \
    tools/gguf_plan.cpp tools/gguf_dma.cpp tools/gguf_mix.cpp tools/gguf_offload.cpp \
    tools/gguf_mirror.cpp -o build/gguf_tool
