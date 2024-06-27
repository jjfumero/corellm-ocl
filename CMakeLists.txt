cmake_minimum_required(VERSION 3.27)
project(corellvms)

set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-lOpenCL")

# Add the kernelPtr files in the bin directory
configure_file(kernels.cl ${CMAKE_BINARY_DIR}/kernels.cl)

add_executable(corellvms main.cpp
)
