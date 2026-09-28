# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

import quadruped_robot.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        self._sample_command_modes(env_ids)
        # Set commands below the 0.2 m/s deadband to zero. Keep exactly 0.2 m/s:
        # that is the first linear command produced by the 20% command curriculum.
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) >= 0.2).unsqueeze(1)

    def _sample_command_modes(self, env_ids: Sequence[int]) -> None:
        """Increase coverage of cardinal commands without changing the default distribution.

        A fully independent uniform sample of ``vx``, ``vy`` and ``wz`` almost
        never produces a pure lateral or pure yaw command.  Those are exactly
        the commands used during teleoperation, so a configuration can reserve
        a fraction of environments for each command family.  All fractions are
        zero by default to preserve the behavior of existing tasks.
        """
        mode_fractions = (
            self.cfg.rel_forward_envs,
            self.cfg.rel_lateral_envs,
            self.cfg.rel_yaw_envs,
            self.cfg.rel_mixed_envs,
        )
        if not any(mode_fractions) or len(env_ids) == 0:
            return
        if sum(mode_fractions) > 1.0 + 1e-6:
            raise ValueError("The sum of command-mode fractions must not exceed one.")

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        commands = self.vel_command_b[env_ids]
        # Preserve standing commands selected by the parent command generator.
        active = torch.linalg.vector_norm(commands, dim=1) > 1e-6
        if not active.any():
            return
        active_ids = env_ids[active]
        random_values = torch.rand(active_ids.numel(), device=self.device)
        boundaries = torch.tensor(mode_fractions, device=self.device).cumsum(dim=0)

        forward_mask = random_values < boundaries[0]
        lateral_mask = (random_values >= boundaries[0]) & (random_values < boundaries[1])
        yaw_mask = (random_values >= boundaries[1]) & (random_values < boundaries[2])
        mixed_mask = (random_values >= boundaries[2]) & (random_values < boundaries[3])

        if forward_mask.any():
            ids = active_ids[forward_mask]
            self.vel_command_b[ids, 0] = self._sample_nonzero_axis(self.cfg.ranges.lin_vel_x, ids.numel())
            self.vel_command_b[ids, 1:] = 0.0
        if lateral_mask.any():
            ids = active_ids[lateral_mask]
            self.vel_command_b[ids, 0] = 0.0
            self.vel_command_b[ids, 1] = self._sample_nonzero_axis(self.cfg.ranges.lin_vel_y, ids.numel())
            self.vel_command_b[ids, 2] = 0.0
        if yaw_mask.any():
            ids = active_ids[yaw_mask]
            self.vel_command_b[ids, :2] = 0.0
            self.vel_command_b[ids, 2] = self._sample_nonzero_axis(self.cfg.ranges.ang_vel_z, ids.numel())
        if mixed_mask.any():
            ids = active_ids[mixed_mask]
            self.vel_command_b[ids, 0] = self._sample_nonzero_axis(self.cfg.ranges.lin_vel_x, ids.numel())
            self.vel_command_b[ids, 1] = self._sample_nonzero_axis(self.cfg.ranges.lin_vel_y, ids.numel())
            self.vel_command_b[ids, 2] = self._sample_nonzero_axis(self.cfg.ranges.ang_vel_z, ids.numel())

    def _sample_nonzero_axis(self, bounds: tuple[float, float], count: int) -> torch.Tensor:
        """Sample an axis while avoiding the near-zero deadband used for commands."""
        low, high = bounds
        if low >= 0.0 or high <= 0.0:
            return torch.empty(count, device=self.device).uniform_(low, high)
        max_abs = max(abs(low), abs(high))
        min_abs = min(0.25, max_abs)
        magnitude = torch.empty(count, device=self.device).uniform_(min_abs, max_abs)
        signs = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        return magnitude * signs

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand
    rel_forward_envs: float = 0.0
    rel_lateral_envs: float = 0.0
    rel_yaw_envs: float = 0.0
    rel_mixed_envs: float = 0.0
