#!/bin/bash
#
# Build and test script for iAODE package
#
# Usage: bash build_and_test.sh

set -e  # Exit on error

echo "======================================================================"
echo "iAODE Package Build and Test"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
echo "Done."
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip setuptools wheel build twine
echo "Done."
echo ""

# Build package
echo "Building package..."
python -m build
echo "Package built successfully!"
echo ""

# Check package
echo "Checking package with twine..."
twine check dist/*
echo "Package check passed!"
echo ""

# List contents
echo "Package contents:"
ls -lh dist/
echo ""

# Test import (local installation)
echo "Testing local installation..."
pip install -e .
echo ""

echo "Testing package import..."
python -c "
import iaode
print(f'✓ iaode version: {iaode.__version__}')
print(f'✓ Available modules: {dir(iaode)}')
print('✓ Package imported successfully!')
"
echo ""

echo "======================================================================"
echo "Build and test completed successfully!"
echo ""
echo "To upload to PyPI:"
echo "  1. Test PyPI: twine upload --repository testpypi dist/*"
echo "  2. Production: twine upload dist/*"
echo ""
echo "To install from local build:"
echo "  pip install dist/iaode-*.whl"
echo "======================================================================"
