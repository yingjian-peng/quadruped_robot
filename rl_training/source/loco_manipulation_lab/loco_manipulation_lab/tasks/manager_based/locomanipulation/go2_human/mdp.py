"""Commands, observations, and rewards for Go2 rear-leg balance training."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass


class HumanBalanceCommand(CommandTerm):
    """Sample the legacy ``vx, vy, vz, roll-rate`` command directly in body coordinates."""

    cfg: "HumanBalanceCommandCfg"

    def __init__(self, cfg: "HumanBalanceCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.metrics["error_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_roll_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_lin_vel"] += torch.linalg.norm(
            self.command_b[:, :3] - self.robot.data.root_lin_vel_b, dim=1
        ) / max_command_step
        self.metrics["error_roll_rate"] += torch.abs(
            self.command_b[:, 3] - self.robot.data.root_ang_vel_b[:, 0]
        ) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        values = torch.empty(len(env_ids), device=self.device)
        self.command_b[env_ids, 0] = values.uniform_(*self.cfg.ranges.lin_vel_x)
        self.command_b[env_ids, 1] = values.uniform_(*self.cfg.ranges.lin_vel_y)
        self.command_b[env_ids, 2] = values.uniform_(*self.cfg.ranges.lin_vel_z)
        self.command_b[env_ids, 3] = values.uniform_(*self.cfg.ranges.roll_rate)

    def _update_command(self):
        pass


@configclass
class HumanBalanceCommandCfg(CommandTermCfg):
    class_type: type = HumanBalanceCommand
    asset_name: str = MISSING
    """Name of the Go2 articulation."""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        lin_vel_z: tuple[float, float] = (-2.0, 2.0)
        roll_rate: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = Ranges()


def human_balance_command(env, command_name: str) -> torch.Tensor:
    """Return the explicit four-dimensional balance command."""
    return env.command_manager.get_command(command_name)


def track_human_lin_vel_exp(env, command_name: str, std: float) -> torch.Tensor:
    """Track the legacy body-frame three-dimensional linear velocity command."""
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :3] - asset.data.root_lin_vel_b), dim=1)
    return torch.exp(-error / (std * std))


def track_human_roll_rate_exp(env, command_name: str, std: float) -> torch.Tensor:
    """Track the body-frame roll-rate component of the balance command."""
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 3] - asset.data.root_ang_vel_b[:, 0])
    return torch.exp(-error / (std * std))


def angular_velocity_yz_l2(env) -> torch.Tensor:
    """Penalize the pitch and yaw components while roll-rate is commanded separately."""
    asset: Articulation = env.scene["robot"]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, 1:3]), dim=1)


def lin_vel_x_l2(env) -> torch.Tensor:
    """Penalize fore-aft body velocity, matching the legacy ``lin_vel_x`` term."""
    asset: Articulation = env.scene["robot"]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def target_projected_gravity_l2(env, target: tuple[float, float, float]) -> torch.Tensor:
    """Keep the body at the specified balance orientation using projected gravity."""
    asset: Articulation = env.scene["robot"]
    target_tensor = torch.tensor(target, dtype=torch.float32, device=asset.device)
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_tensor), dim=1)


def rear_single_contact(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward one rear-foot contact whenever a non-zero balance command is active."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold
    command = env.command_manager.get_command(command_name)
    return (torch.sum(contacts, dim=1) == 1).float() * (torch.linalg.norm(command[:, :3], dim=1) > 0.05)


def hip_joint_abs_sum(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Legacy sum of absolute hip positions for the four leg chains."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def rear_hip_symmetry_l1(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Legacy RL/RR hip symmetry cost."""
    asset: Articulation = env.scene[asset_cfg.name]
    joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.abs(joints[:, 0] - joints[:, 1])


def relative_height_scan(env, sensor_cfg: SceneEntityCfg, offset: float, scale: float) -> torch.Tensor:
    """Return the legacy clipped root-height terrain observation."""
    sensor = env.scene[sensor_cfg.name]
    robot: Articulation = env.scene["robot"]
    heights = robot.data.root_pos_w[:, 2].unsqueeze(1) - offset - sensor.data.ray_hits_w[..., 2]
    return torch.clamp(heights, -1.0, 1.0) * scale
