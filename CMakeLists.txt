project(matmul LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.2)

set(CUDA_NVCC_FLAGS -O3;-G;-g)
add_executable(matmul main.cu)
target_link_libraries(matmul PUBLIC -lcublas -lcurand PRIVATE nvToolsExt)