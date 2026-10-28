g++ -std=c++23 -O2 -Icore -Imodels/base -Imemory -Imodels/graph -Ibackends/ggml tools/dump_expert_bytes.cpp -o /tmp/dump_expert_bytes
/tmp/dump_expert_bytes <your model.gguf path> 0 199 gate /tmp/e199_gate.bin
