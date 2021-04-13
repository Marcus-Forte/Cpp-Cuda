cmake_minimum_required(VERSION 3.8)
project(cmake_cuda LANGUAGES CUDA CXX)

find_package(Eigen3 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CUDA_FLAGS "-Wall")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${EIGEN3_INCLUDE_DIRS}) # runtime.h

add_executable(math_test src/main.cpp src/cuda_math.cu)
add_executable(matrix_inv src/matrix_inv.cu)

target_link_libraries(matrix_inv cublas)




