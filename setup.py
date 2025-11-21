#!/usr/bin/env python
"""
Setup configuration for iaode package.
This file is kept for backward compatibility. Modern build uses pyproject.toml.
"""

from setuptools import setup  # type: ignore

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Main setup configuration is in pyproject.toml
# This file is kept for backward compatibility
setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
)
