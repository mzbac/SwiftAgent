#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build using xcodebuild with the Package.swift directly
echo "Building SwiftAgentCLI with xcodebuild..."
echo "This is required for MLX Metal shader support..."

# Build with xcodebuild for Metal shader support
xcodebuild -scheme SwiftAgentCLI \
           -destination "platform=macOS,arch=arm64" \
           build

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Find the built executable
# xcodebuild puts it in DerivedData, let's find it
EXECUTABLE=$(xcodebuild -scheme SwiftAgentCLI -showBuildSettings | grep -m 1 "BUILT_PRODUCTS_DIR" | grep -oE '/.*' | sed 's/[[:space:]]*$//')/SwiftAgentCLI

if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at: $EXECUTABLE"
    echo "Trying to find it..."
    EXECUTABLE=$(find ~/Library/Developer/Xcode/DerivedData -name SwiftAgentCLI -type f 2>/dev/null | grep -v ".dSYM" | head -1)
    if [ -z "$EXECUTABLE" ]; then
        echo "Could not find executable"
        exit 1
    fi
fi

# Run the CLI
echo ""
echo "Running SwiftAgentCLI..."
echo "Executable: $EXECUTABLE"
echo ""
"$EXECUTABLE"