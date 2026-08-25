"""Go2 rear-leg balance and inverted-pose task registration."""

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import Go2RearLegBalancePPORunnerCfg
from .flat_env_cfg import Go2RearLegBalanceEnvCfg


gym.register(
    id="Go2-RearLeg-Balance-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": Go2RearLegBalanceEnvCfg, "rsl_rl_cfg_entry_point": Go2RearLegBalancePPORunnerCfg},
)
