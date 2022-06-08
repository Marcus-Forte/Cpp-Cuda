cmake_minimum_required(VERSION 3.10)
project(cuda-tests LANGUAGES CUDA CXX)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARIES})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CUDA_FLAGS "-Wall")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g")

add_compile_options("-march=native")


include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${EIGEN3_INCLUDE_DIRS}) # runtime.h

add_executable(math_test src/main.cpp src/cuda_math.cu)
add_executable(matrix_inv src/matrix_inv.cu)

add_executable(use_thrust src/use_thrust.cpp)
set_source_files_properties(src/use_thrust.cpp PROPERTIES LANGUAGE CUDA)

add_executable(use_thrust_with_eigen_pcl src/use_thrust_with_eigen_pcl.cpp)
set_source_files_properties(src/use_thrust_with_eigen_pcl.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(matrix_inv cublas)
target_link_libraries(use_thrust_with_eigen_pcl cublas)




