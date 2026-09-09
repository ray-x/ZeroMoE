#!/usr/bin/env bash
# setup.sh — MoEStream Engine Installation and Automated Build Script
set -e

echo "  MoEStream Engine — Setup and Compilation"
echo "----------------------------------------"

# 1. Dependency Toolchain Verification
echo "[1/4] Checking system compiler dependencies..."
command -v cmake >/dev/null || { echo "  Error: cmake is missing."; exit 1; }
command -v g++ >/dev/null || { echo "  Error: g++ is missing."; exit 1; }
command -v python3 >/dev/null || { echo "  Error: python3 is missing."; exit 1; }

# macOS has no NVIDIA CUDA SDK; CPU-only builds are the only option there.
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  -> macOS detected; skipping nvcc check (CPU-only build)"
else
    command -v nvcc >/dev/null || { echo "  Error: NVIDIA CUDA Compiler (nvcc) is missing."; exit 1; }
    echo "  -> nvcc: $(nvcc --version | grep release | awk '{print $5, $6}')"
fi

echo "  -> cmake: $(cmake --version | head -n 1)"
echo "  -> g++: $(g++ -dumpversion)"

# 2. Python Environment Setup for API Gateway
echo "[2/4] Setting up Python virtual environment for Anthropic API..."
if [ ! -d "api/venv" ]; then
    python3 -m venv api/venv
    source api/venv/bin/activate
    if [ ! -f "api/requirements.txt" ]; then
        echo -e "fastapi>=0.100.0\nuvicorn>=0.22.0\nsse-starlette>=1.6.0\npydantic>=2.0.0" > api/requirements.txt
    fi
    api/venv/bin/pip install --quiet --upgrade pip
    api/venv/bin/pip install --quiet -r api/requirements.txt
    echo "  -> Python virtual environment created and packages installed successfully."
else
    echo "  -> Python virtual environment already exists. Skipping environment creation."
fi # Fixed missing fi statements

# 3. C++/CUDA Binary Compilation via CMake
echo "[3/4] Triggering CMake build pipeline..."
mkdir -p build
cd build

# Full engine builds require ggml.  Fetch a stable llama.cpp release by default
# unless the caller opts out (e.g. they provide their own ggml install).
GGML_FLAG="${MOESTREAM_USE_GGML:-${MOESTREAM_USE_LLAMA_CPP:-ON}}"
cmake -DCMAKE_BUILD_TYPE=Release -DMOESTREAM_USE_GGML="$GGML_FLAG" ..
echo "[4/4] Compiling high-performance target binary..."
make -j$(nproc)
cd ..

echo "----------------------------------------"
echo "  MoEStream compilation complete!"
echo "  -> Engine Core Executable: ./build/moestream"
echo "  -> API Gateway: ./api/api_server.py"
echo ""
echo "To serve the Anthropic API endpoint, execute:"
echo "  python3 api/api_server.py"