cmake_minimum_required(VERSION 3.10)
project(inception_v3_hailo)

set(CMAKE_CXX_STANDARD 14)

# Find HailoRT package
find_package(HailoRT REQUIRED)

# Add executable
add_executable(inception_v3_hailo
    main.cpp
    inception_v3_hailortpp.cpp
)

# Include directories
target_include_directories(inception_v3_hailo PRIVATE ${HAILORT_INCLUDE_DIRS})

# Link libraries
target_link_libraries(inception_v3_hailo PRIVATE ${HAILORT_LIBRARIES})