# Top-level convenience wrapper around the CMake build (see CLAUDE.md /
# setup.sh for the full manual recipe). Requires Linux + CUDA + AVX-512 --
# this will not configure or build on macOS.
BUILD_DIR  ?= build
BUILD_TYPE ?= Release
JOBS       := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all build test check clean configure test-components bench-io gguf-tool

all: build

# Re-runs only if CMakeLists.txt changed or the build dir doesn't exist yet --
# cmake itself is idempotent, but this avoids a needless reconfigure on every
# `make` invocation.
$(BUILD_DIR)/CMakeCache.txt: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

configure: $(BUILD_DIR)/CMakeCache.txt

# Main engine binary: ./build/moestream
build: configure
	cmake --build $(BUILD_DIR) --target moestream -j$(JOBS)

# Native test harness (src/main_test.cpp): builds moestream_tests and runs it
# directly so every [Test N] line prints regardless of pass/fail (ctest's
# default swallows stdout on success). Exit code propagates, so `make test`
# fails the same way any other failing recipe does.
#
# Depends on `build` (not just `configure`) so `make test` also rebuilds the
# actual engine binary, ./build/moestream -- moestream and moestream_tests
# are separate CMake targets built from mostly-shared sources, and this
# target used to only build moestream_tests. That let the test harness
# report a change as verified-correct while ./build/moestream, the binary
# real generation runs actually use, silently stayed stale: a real
# AVX-512 dequant fix passed Test 6 here but the unrebuilt ./build/moestream
# still showed the old (slow) behavior in production runs, with nothing in
# `make test`'s own output to suggest why.
test: build
	cmake --build $(BUILD_DIR) --target moestream_tests -j$(JOBS)
	./$(BUILD_DIR)/moestream_tests

check: test

# CPU-only component tests (tests/test_components.cpp) -- deliberately NOT
# routed through CMake/CUDA, so this target runs on macOS too (the workflow:
# test a SUBSET here, run the full `make test` CUDA suite on Linux). Compiles
# the single translation unit with the system C++ compiler in seconds.
# Optional substring filters: make test-components ARGS="rope dequant".
test-components:
	tests/run_components.sh $(ARGS)

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

clean:
	rm -rf $(BUILD_DIR)
