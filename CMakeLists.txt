cmake_minimum_required(VERSION 3.10)
project(RenderEngine)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 26)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect all source files from the src directory
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Add a target for the library (STATIC or SHARED)
add_library(RenderEngine STATIC ${SOURCES})

# Optionally, create an install target
install(TARGETS RenderEngine DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
