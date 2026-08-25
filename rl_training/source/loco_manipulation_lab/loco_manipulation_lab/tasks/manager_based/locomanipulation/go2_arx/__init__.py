"""Go2-ARX locomotion-manipulation task registration."""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import Go2ArxFlatPPORunnerCfg
from .flat_env_cfg import Go2ArxFlatEnvCfg


gym.register(
    id="Go2-Arx-LocoManip-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2ArxFlatEnvCfg,
        "rsl_rl_cfg_entry_point": Go2ArxFlatPPORunnerCfg,
    },
)
