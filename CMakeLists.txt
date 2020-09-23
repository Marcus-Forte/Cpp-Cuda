cmake_minimum_required(VERSION 3.8)
project(cmake_cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "-Wall")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g")

add_executable(cuda_exe src/main.cu src/vectorAdd.cpp src/vectorAdd.cu)

set_target_properties(cuda_exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_exe PROPERTY CUDA_ARCHITECTURES 35)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
