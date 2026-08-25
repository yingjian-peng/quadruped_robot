"""Go2W rough-terrain velocity training task registration."""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import Go2WRoughPPORunnerCfg
from .rough_env_cfg import Go2WRoughEnvCfg


gym.register(
    id="Go2W-WheelQuadruped-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2WRoughEnvCfg, "rsl_rl_cfg_entry_point": Go2WRoughPPORunnerCfg},
)
