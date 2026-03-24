cmake_minimum_required(VERSION 3.20)
project(rais LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Header-only library target for now
add_library(rais INTERFACE)
target_include_directories(rais INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Catch2
find_package(Catch2 3 REQUIRED)

# Tests
enable_testing()

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_queue COMMAND test_queue)

# Benchmarks
add_executable(bench_queue benchmarks/bench_queue.cpp)
target_link_libraries(bench_queue PRIVATE rais)
