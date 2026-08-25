"""RSL-RL PPO configuration migrated from the legacy Go2W runner."""

from isaaclab.utils import configclass

from ...b2w.agents.rsl_rl_ppo_cfg import B2WRoughPPORunnerCfg


@configclass
class Go2WRoughPPORunnerCfg(B2WRoughPPORunnerCfg):
    experiment_name = "go2w_isaaclab_rough"
