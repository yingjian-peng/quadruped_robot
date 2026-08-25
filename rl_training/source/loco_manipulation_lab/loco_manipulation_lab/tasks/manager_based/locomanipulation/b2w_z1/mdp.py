"""Observation and manipulation-reward terms for the B2W-Z1 task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def joint_pos_rel_without_wheels(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return default-relative 22D joint positions with wheel angles masked."""
    asset: Articulation = env.scene[asset_cfg.name]
    result = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    result[:, 12:16] = 0.0
    return result


def end_effector_position_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected end-effector position in the robot root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if body_pos_w.ndim == 3:
        body_pos_w = body_pos_w[:, 0]
    return quat_apply_inverse(asset.data.root_quat_w, body_pos_w - asset.data.root_pos_w)


def pose_command_position(env, command_name: str) -> torch.Tensor:
    """Return the Cartesian component of an Isaac Lab pose command."""
    return env.command_manager.get_command(command_name)[:, :3]


def pose_command_position_delta(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return local end-effector displacement from the current Cartesian target."""
    return end_effector_position_b(env, asset_cfg) - pose_command_position(env, command_name)


def pose_command_position_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return squared local Cartesian end-effector error."""
    error = pose_command_position_delta(env, command_name, asset_cfg)
    return torch.sum(torch.square(error), dim=-1)


def pose_command_position_exp(env, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Exponentiated local Cartesian tracking reward."""
    return torch.exp(-pose_command_position_error(env, command_name, asset_cfg) / (std * std))


def relative_height_scan(env, sensor_cfg: SceneEntityCfg, offset: float, scale: float) -> torch.Tensor:
    """Match the legacy clipped root-height terrain observation."""
    sensor = env.scene[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]
    heights = robot.data.root_pos_w[:, 2].unsqueeze(1) - offset - sensor.data.ray_hits_w[..., 2]
    return torch.clamp(heights, -1.0, 1.0) * scale
