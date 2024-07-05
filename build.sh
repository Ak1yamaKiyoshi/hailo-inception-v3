#!/bin/bash

# Set the project directory name
PROJECT_DIR="."

# Get the build mode from the command line (default to release)
if [ "$1" = "debug" ]; then
    BUILD_MODE="Debug"
else
    BUILD_MODE="Release"
fi

# Create the build directory
BUILD_DIR="$PROJECT_DIR/build.$BUILD_MODE"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure the project with CMake
cmake -DCMAKE_BUILD_TYPE=$BUILD_MODE ..

# Compile the project
make

# Return to the original directory
cd ..