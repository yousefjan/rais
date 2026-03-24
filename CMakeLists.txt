cmake_minimum_required(VERSION 3.20)
project(rais LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Platform detection: enable Objective-C++ and Metal on Apple
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
    set(RAIS_HAS_METAL ON)
else()
    set(RAIS_HAS_METAL OFF)
endif()

# Library target
add_library(rais STATIC src/scheduler.cpp)
target_include_directories(rais PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Metal shader compilation: .metal → .air → .metallib
if(RAIS_HAS_METAL)
    set(METAL_SHADER_SOURCE ${CMAKE_SOURCE_DIR}/shaders/rais_kernels.metal)
    set(METAL_AIR ${CMAKE_BINARY_DIR}/rais_kernels.air)
    set(METAL_LIB ${CMAKE_BINARY_DIR}/rais_kernels.metallib)

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SHADER_SOURCE} -o ${METAL_AIR}
        DEPENDS ${METAL_SHADER_SOURCE}
        COMMENT "Compiling Metal shaders to AIR"
    )

    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking AIR to Metal library"
    )

    add_custom_target(metal_shaders DEPENDS ${METAL_LIB})
endif()

# Catch2
find_package(Catch2 3 REQUIRED)

# Tests
enable_testing()

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_queue COMMAND test_queue)

add_executable(test_deque tests/test_deque.cpp)
target_link_libraries(test_deque PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_deque COMMAND test_deque)

# Benchmarks
add_executable(bench_queue benchmarks/bench_queue.cpp)
target_link_libraries(bench_queue PRIVATE rais)

add_executable(bench_deque benchmarks/bench_deque.cpp)
target_link_libraries(bench_deque PRIVATE rais)

add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE rais Catch2::Catch2WithMain)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(bench_scheduler benchmarks/bench_scheduler.cpp)
target_link_libraries(bench_scheduler PRIVATE rais)
