cmake_minimum_required(VERSION 3.0.0)
project(neural-net VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(neural-net main.cpp src/n-net.cpp src/neuron.cpp lib/lodepng.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

include(CPack)
