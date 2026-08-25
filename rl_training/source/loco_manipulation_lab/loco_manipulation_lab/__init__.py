"""Isaac Lab implementation of the quadruped locomotion training project."""

from pathlib import Path


QUADRUPED_ROBOT_ROOT_DIR = Path(__file__).resolve().parents[4]
"""Root directory of the quadruped_robot monorepo."""

LOCO_MANIPULATION_LAB_ROOT_DIR = QUADRUPED_ROBOT_ROOT_DIR / "rl_training"
"""Root directory of the training subproject."""

ROBOT_MODELS_DIR = QUADRUPED_ROBOT_ROOT_DIR / "robot_models"
"""Shared robot-model directory used by training and deployment."""
