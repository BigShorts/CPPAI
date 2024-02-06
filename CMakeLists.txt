cmake_minimum_required(VERSION 3.5)
project(CPPAI)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")
add_executable(CPPAI ${SRC_FILES})

set(LLAMA_BLAS ON)
set(LLAMA_BLAS_VENDOR OpenBLAS)
add_subdirectory(lib/llama.cpp)
target_link_libraries(CPPAI llama)
target_include_directories(CPPAI PUBLIC lib/llama.cpp/include)
