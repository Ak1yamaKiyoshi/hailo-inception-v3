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

# Build the project and show full output
cmake --build . -- VERBOSE=1

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Build failed. Please check the error messages above."
    exit 1
fi

# Copy the shared library and executable to the parent directory
cp libinception_v3_inference.so ..
cp inception_v3_inference ..

# Return to the original directory
cd ..

# Remove HailoRT log if it exists
if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi

echo "Build complete. Shared library and executable are now in the current directory."