"""B2W rough-terrain velocity training task registration."""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import B2WRoughPPORunnerCfg
from .rough_env_cfg import B2WRoughEnvCfg


gym.register(
    id="B2W-WheelQuadruped-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": B2WRoughEnvCfg, "rsl_rl_cfg_entry_point": B2WRoughPPORunnerCfg},
)
