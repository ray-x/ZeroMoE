cmake_minimum_required(VERSION 3.18)
project(MoEStreamTools LANGUAGES CXX)

# The parent CMakeLists sets the global C++ standard; do not override it here.
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 23)
endif()

add_executable(gguf_tool
    gguf_tool.cpp
    gguf_pack.cpp
    gguf_io.cpp
    gguf_verify.cpp
    gguf_plan.cpp
    gguf_dma.cpp
    gguf_mix.cpp
    gguf_offload.cpp
)

add_executable(exl3_tool
    exl3_tool.cpp
    ../models/base/safetensors_adapter.cpp
)
target_include_directories(exl3_tool PRIVATE ../include)
set_target_properties(exl3_tool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
target_include_directories(gguf_tool PRIVATE ../include)

# Put the binary at the build root so `make build` and the shell wrapper both
# produce ./build/gguf_tool.
set_target_properties(gguf_tool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gguf_tool PRIVATE OpenMP::OpenMP_CXX)
endif()
