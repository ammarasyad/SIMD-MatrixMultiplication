cmake_minimum_required(VERSION 3.24)
project(Matdis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native -mtune=native -mavx2 -mfma")

add_executable(Matdis main.cpp)
