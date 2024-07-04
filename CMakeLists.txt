cmake_minimum_required(VERSION 3.10.0)
project(inception_v3_inference)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

set(COMPILE_OPTIONS_CPP -Wall -Werror -O3)

file(GLOB SOURCES ./*.cpp)

# Create the executable
add_executable(${PROJECT_NAME}_exe ${SOURCES})

# Create the shared library
add_library(${PROJECT_NAME}_lib SHARED ${SOURCES})

include_directories(/usr/include ./)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set compile options and link libraries for the executable
target_compile_options(${PROJECT_NAME}_exe PRIVATE ${COMPILE_OPTIONS_CPP})
target_link_libraries(${PROJECT_NAME}_exe HailoRT::libhailort Threads::Threads ${OpenCV_LIBS})

# Set compile options and link libraries for the shared library
target_compile_options(${PROJECT_NAME}_lib PRIVATE ${COMPILE_OPTIONS_CPP})
target_link_libraries(${PROJECT_NAME}_lib HailoRT::libhailort Threads::Threads ${OpenCV_LIBS})

# Set the output name of the executable
set_target_properties(${PROJECT_NAME}_exe PROPERTIES OUTPUT_NAME "inception_v3_inference")

# Set the output name of the shared library
set_target_properties(${PROJECT_NAME}_lib PROPERTIES OUTPUT_NAME "inception_v3_inference")
