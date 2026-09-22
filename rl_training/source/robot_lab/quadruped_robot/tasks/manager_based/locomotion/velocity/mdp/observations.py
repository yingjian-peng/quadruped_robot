# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Observations specific to the locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float, command_name: str) -> torch.Tensor:
    """Gait phase clock as a sin/cos pair (2-dim), zeroed when standing.

    Ported from Fysics_rl_mjlab (unitree_rl_mjlab) ``mdp.phase``: the phase wraps every
    ``period`` seconds and restarts at each episode reset, giving the actor an internal
    oscillator signal to entrain a periodic gait (e.g. trot).

    Args:
        period: Duration of one full gait cycle in seconds.
        command_name: Name of the command term; the phase is zeroed while the commanded
            velocity is (near) zero, i.e. standing.
    """
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
