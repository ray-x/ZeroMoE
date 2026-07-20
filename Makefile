# Top-level convenience wrapper around the CMake build (see CLAUDE.md /
# setup.sh for the full manual recipe). Requires Linux + CUDA + AVX-512 --
# this will not configure or build on macOS.
BUILD_DIR  ?= build
BUILD_TYPE ?= Release
JOBS       := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all build test check clean configure

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

clean:
	rm -rf $(BUILD_DIR)
