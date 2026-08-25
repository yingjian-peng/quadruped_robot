"""Installation metadata for the Isaac Lab implementation."""

from setuptools import find_packages, setup


setup(
    name="loco_manipulation_lab",
    version="0.1.0",
    description="Isaac Lab 4.5 environments for legged robot manipulation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
)
