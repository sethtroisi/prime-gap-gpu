cmake_minimum_required(VERSION 3.16)
project(prime-gap-gpu)

set(CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/cmake
    ${CMAKE_MODULE_PATH}
)

# TODO are these three lines needed?
find_package(GMP REQUIRED)
message(STATUS "GMP_INCLUDES: ${GMP_INCLUDES} ${GMP_LIBRARIES}")
link_libraries(${GMP_LIBRARIES})

link_libraries(gmp)

set(CMAKE_CXX_STANDARD 14)

add_executable(gap_search       sieve.cpp gap_common.cpp)

target_link_libraries(combined_sieve primesieve)
