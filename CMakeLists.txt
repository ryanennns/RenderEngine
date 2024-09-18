cmake_minimum_required(VERSION 3.10)
project(RenderEngine LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${SDL2_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GTest REQUIRED)

file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/renderEngine*.cpp)

add_executable(
        RenderEngine
        main.cpp
        cuda/kernels.cpp
        cuda/kernels.cu
        src/components/Vector3D.cpp
        src/renderEngine/ViewPort.cpp
        cuda/helpers.cu
)

set_target_properties(RenderEngine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(RenderEngine CUDA::cudart ${SDL2_LIBRARIES})

add_executable(
        test_suite
        cuda/helpers.cu
        cuda/kernels.cu
        cuda/kernels.cpp
        src/renderEngine/ViewPort.cpp
        src/components/Vector3D.cpp

        tests/LineTest.cu
        tests/VectorTest.cu
        tests/TriangleTest.cu
        tests/IntegrationTest.cu
        tests/LandscapeTest.cu
)

target_link_libraries(test_suite GTest::gtest GTest::gtest_main pthread)

add_test(NAME RenderEngineTests COMMAND test_suite)
