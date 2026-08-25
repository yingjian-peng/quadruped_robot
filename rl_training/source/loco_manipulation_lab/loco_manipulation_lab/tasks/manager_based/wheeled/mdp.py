"""Observation and reward terms specific to the B2W and Go2W tasks."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def base_height_command(env, target_height: float) -> torch.Tensor:
    """Return the legacy scalar base-height command for every environment."""
    return torch.full((env.num_envs, 1), target_height, dtype=torch.float32, device=env.device)


def joint_pos_rel_without_wheels(env, asset_cfg: SceneEntityCfg, wheel_count: int = 4) -> torch.Tensor:
    """Return default-relative joint positions while masking unbounded wheel angles."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    result = joint_pos - default_pos
    result[:, -wheel_count:] = 0.0
    return result


def joint_pos_without_wheels(env, asset_cfg: SceneEntityCfg, wheel_count: int = 4) -> torch.Tensor:
    """Return absolute joint positions while masking unbounded wheel angles."""
    asset: Articulation = env.scene[asset_cfg.name]
    result = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()
    result[:, -wheel_count:] = 0.0
    return result


def relative_height_scan(env, sensor_cfg: SceneEntityCfg, offset: float, scale: float) -> torch.Tensor:
    """Match the legacy ``clip(base_z - offset - measured_height)`` term."""
    sensor = env.scene[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]
    heights = robot.data.root_pos_w[:, 2].unsqueeze(1) - offset - sensor.data.ray_hits_w[..., 2]
    return torch.clamp(heights, -1.0, 1.0) * scale


def stand_still_leg_pos_l1(
    env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """Penalize leg displacement, but never wheel rotation, for a zero command."""
    command = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    return error * (torch.linalg.norm(command[:, :2], dim=1) < command_threshold)


def hip_action_l2(env, action_name: str = "leg_joint_pos") -> torch.Tensor:
    """Legacy hip-action L2 penalty for leg ABI indices 0, 3, 6, and 9."""
    action = env.action_manager.get_term(action_name).raw_actions
    if action.shape[1] != 12:
        raise ValueError(f"Expected 12 leg actions for {action_name!r}, received {action.shape[1]}.")
    return torch.sum(torch.square(action[:, (0, 3, 6, 9)]), dim=1)
