# Top-level convenience wrapper around the CMake build (see CLAUDE.md /
# setup.sh for the full manual recipe).
#
# On Linux with the NVIDIA CUDA Toolkit, this builds the full engine and the
# native CUDA test harness.  On macOS the root CMakeLists.txt automatically
# switches to ZEROMOE_CPU_ONLY=ON and builds the tools plus the CPU-only
# component tests (no CUDA required).  The CPU-only path also works on Linux
# by passing ZEROMOE_CPU_ONLY=ON to CMake manually.
BUILD_TYPE ?= Release
JOBS       := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Use separate build directories for the two supported host platforms so that
# switching between a Linux+CUDA machine and a macOS machine (or sharing the
# same checkout across both) never mixes incompatible CMake caches.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BUILD_DIR ?= build-macos
else
    BUILD_DIR ?= build
endif

.PHONY: all build test check clean clean-all configure reconfigure test-components test-settings inventory-settings bench-io bench-exl3 bench-exl3-cpu bench-exl3-hc gguf-tool qwen4exp-pointer-spike bench-pointer-spike qwen4exp-staging-spike bench-staging-spike qwen-cpu-offload-graph-sandbox bench-cpu-offload-graph-sandbox

all: build

# Re-runs only if CMakeLists.txt changed or the build dir doesn't exist yet --
# cmake itself is idempotent, but this avoids a needless reconfigure on every
# `make` invocation.
$(BUILD_DIR)/CMakeCache.txt: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

configure: $(BUILD_DIR)/CMakeCache.txt

# Remove the current platform's build dir and reconfigure from scratch. Useful
# after switching branches, CMake options, or host platforms.
reconfigure: clean
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

# Default build: all targets CMake knows about for this platform.
# On Linux+CUDA that is zeromoe + zeromoe_tests + gguf_tool.
# On macOS that is test_components + gguf_tool.
build: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

# Native test harness: builds and runs whichever test binary CMake produced
# for this platform. On Linux+CUDA that is zeromoe_tests; on macOS/CPU-only
# that is test_components. Exit code propagates, so `make test` fails the same
# way any other failing recipe does.
test: build
	@if [ -f "$(BUILD_DIR)/zeromoe_tests" ]; then \
		cmake --build $(BUILD_DIR) --target zeromoe_tests -j$(JOBS); \
		./$(BUILD_DIR)/zeromoe_tests; \
	elif [ -f "$(BUILD_DIR)/test_components" ]; then \
		cmake --build $(BUILD_DIR) --target test_components -j$(JOBS); \
		./$(BUILD_DIR)/test_components; \
	else \
		echo "No test binary found in $(BUILD_DIR)"; exit 1; \
	fi

check: test

# CPU-only component tests (tests/test_components.cpp) -- deliberately NOT
# routed through CMake/CUDA, so this target runs on macOS too (the workflow:
# test a SUBSET here, run the full `make test` CUDA suite on Linux). Compiles
# the single translation unit with the system C++ compiler in seconds.
# Optional substring filters: make test-components ARGS="rope dequant".
test-components:
	tests/run_components.sh $(ARGS)

test-settings: configure
	cmake --build $(BUILD_DIR) --target graph_settings_test graph_settings_binding_test -j$(JOBS)
	ctest --test-dir $(BUILD_DIR) -R '^graph_settings' --output-on-failure
	python3 tools/config_inventory.py --check

inventory-settings:
	python3 tools/config_inventory.py $(ARGS)

# gguf_tool (tools/gguf_tool.cpp + gguf_pack/io/verify.cpp) -- cross-platform
# (macOS + Linux), no CMake/CUDA. One binary, three subcommands: pack, io,
# verify. Pass the subcommand + its args via ARGS, e.g.
#   make gguf-tool ARGS="verify ./repacked"
#   make gguf-tool ARGS="pack model.gguf ./repacked --verify-all"
gguf-tool:
	tools/run_gguf_tool.sh $(ARGS)

# I/O benchmark, repurposed onto gguf_tool's `io` subcommand -- A/Bs
# per-expert read pattern: original packed layout vs repacked tight layout.
# Pass orig + repacked paths (and optional --reads/--seed/--direct) via ARGS,
# e.g. make bench-io ARGS="model.gguf ./repacked --reads 5000".
bench-io:
	tools/run_gguf_tool.sh io $(ARGS)

bench-exl3: configure
	cmake --build $(BUILD_DIR) --target exl3_kernel_bench -j$(JOBS)
	./$(BUILD_DIR)/exl3_kernel_bench $(ARGS)

bench-exl3-cpu: configure
	cmake --build $(BUILD_DIR) --target exl3_cpu_bench -j$(JOBS)
	./$(BUILD_DIR)/exl3_cpu_bench $(ARGS)

bench-exl3-hc: configure
	cmake --build $(BUILD_DIR) --target exl3_hc_bench -j$(JOBS)
	./$(BUILD_DIR)/exl3_hc_bench $(ARGS)

# Layer-13 Phase-1 pointer-indirection laboratory.  This is deliberately
# separate from `make test`: it requires CUDA host mapping and reports live
# VRAM/mapped/H2D-stage measurements without touching engine dispatch.
qwen4exp-pointer-spike: configure
	cmake --build $(BUILD_DIR) --target qwen4exp_pointer_spike -j$(JOBS)

bench-pointer-spike: qwen4exp-pointer-spike
	./$(BUILD_DIR)/qwen4exp_pointer_spike $(ARGS)

# Layer-13 Phase-2 device-resident staging + graph gather laboratory.
qwen4exp-staging-spike: configure
	cmake --build $(BUILD_DIR) --target qwen4exp_staging_spike -j$(JOBS)

bench-staging-spike: qwen4exp-staging-spike
	./$(BUILD_DIR)/qwen4exp_staging_spike $(ARGS)

# CPUOffload Milestone-C synthetic CUDA-graph host-node sandbox. Arguments are:
# iterations elements cpu_repeats gpu_work gpu_repeats.
qwen-cpu-offload-graph-sandbox: configure
	cmake --build $(BUILD_DIR) --target qwen_cpu_offload_graph_sandbox -j$(JOBS)

bench-cpu-offload-graph-sandbox: qwen-cpu-offload-graph-sandbox
	./$(BUILD_DIR)/qwen_cpu_offload_graph_sandbox $(ARGS)

clean:
	rm -rf $(BUILD_DIR)

clean-all:
	rm -rf build build-macos build-*
