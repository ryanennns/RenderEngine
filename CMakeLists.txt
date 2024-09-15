cmake_minimum_required(VERSION 3.10)
project(RenderEngine LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# CUDA setup
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # Enable separable compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")  # Ensure RDC is enabled

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Collect all source files from the src directory
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/renderEngine*.cpp)

# Add the executable and ensure both kernels.cpp and kernels.cu are compiled and linked
add_executable(
        test
        main.cpp
        cuda/kernels.cpp  # C++ wrapper
        cuda/kernels.cu   # CUDA kernel implementation
        src/components/Vector3D.cpp
        src/components/Ray.cpp
        src/renderEngine/ViewPort.cpp
)

# Ensure that device linking is enabled for the final executable
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the CUDA runtime and other necessary libraries
target_link_libraries(test CUDA::cudart ${SDL2_LIBRARIES})
