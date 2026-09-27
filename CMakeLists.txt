cmake_minimum_required(VERSION 3.16)
project(prime-gap-gpu LANGUAGES CXX CUDA)

# --std=c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
# -sm_89
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/cmake
    ${CMAKE_MODULE_PATH}
)

#set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=-*,bugprone-*,modernize-*")
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

set(BITS 512 CACHE STRING "Bit size definition for GPU search")

find_package(CUDAToolkit REQUIRED)
find_library(GMP_LIB gmp REQUIRED)
find_library(PRIMESIEVE_LIB primesieve REQUIRED)

link_libraries(gmp)
link_libraries(primesieve)

include_directories(
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../CGBN/include
)

add_library(common INTERFACE)

target_compile_options(common
    INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Werror
            -Wno-vla
            -mtune=native
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            --resource-usage
            -Xcompiler=-Wall,-Werror
            -mtune=native
            -DGPU_BITS=${BITS}
            -DMP=${MP}
        >
)

add_executable(gap_search_primorial
        gap_search_primorial.cpp
                gap_common.cpp gap_search_common.cpp
                overflow.cpp
                xoroshiro128plus.cpp
                gpu_sieve.cu gpu_testing.cu)
target_link_libraries(gap_search_primorial PRIVATE common)

add_executable(gpu_benchmark
        gpu_benchmark.cu
                gap_common.cpp)
target_link_libraries(gpu_benchmark PRIVATE common)
