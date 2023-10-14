cmake_minimum_required(VERSION 3.0.0)
project(neural-net-c VERSION 0.1.0 LANGUAGES C)

include(CTest)
enable_testing()

add_executable(neural-net-c main.c src/net.c src/neuron.c)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")


find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

target_link_libraries(neural-net-c m ${PNG_LIBRARY})

include(CPack)
