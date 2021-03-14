cmake_minimum_required(VERSION 3.16)
project(OpenCvVideoStab)

set(CMAKE_CXX_STANDARD 11)

add_executable(OpenCvVideoStab main.cpp)
add_executable(OpenCvVideoStab naive/naive.cpp)