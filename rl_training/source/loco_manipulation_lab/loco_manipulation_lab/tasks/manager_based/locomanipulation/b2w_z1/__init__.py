"""B2W-Z1 locomotion-manipulation task registration."""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import B2WZ1FlatPPORunnerCfg
from .flat_env_cfg import B2WZ1FlatEnvCfg


gym.register(
    id="B2W-Z1-LocoManip-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": B2WZ1FlatEnvCfg, "rsl_rl_cfg_entry_point": B2WZ1FlatPPORunnerCfg},
)
