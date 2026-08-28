"""Installation metadata for the focused Isaac Lab training extension."""

from setuptools import find_packages, setup


setup(
    name="quadruped_robot_lab",
    version="0.1.0",
    description="Isaac Lab 4.5 locomotion training for DeepRobotics and Unitree robots",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
)
