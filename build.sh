#!/bin/bash

# Remove old build directory if it exists
if [ -d "build" ]; then
    rm -rf build
fi

# Create new build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# Copy the shared library to the parent directory
cp libinception_v3_inference.so ..

# Return to the original directory
cd ..

# Remove HailoRT log if it exists
if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi

echo "Build complete. Shared library is now in the current directory."