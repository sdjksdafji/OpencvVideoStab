cmake_minimum_required(VERSION 3.16)
project(OpenCvVideoStab)

set(CMAKE_CXX_STANDARD 11)
set(Sources cuda/cuda.cpp)
add_executable(OpenCvVideoStab main.cpp)
add_executable(OpenCvVideoStabCuda cuda/cuda.cpp)