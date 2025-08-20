#!/bin/bash
# Run type checking with ty

echo "=== Fluid Server Type Check ==="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

echo "Installing ty..."
uv add --dev ty

echo ""
echo "Running type check with ty..."
uv run ty

if [ $? -eq 0 ]; then
    echo ""
    echo "Type check passed!"
else
    echo ""
    echo "Type check failed!"
    exit 1
fi