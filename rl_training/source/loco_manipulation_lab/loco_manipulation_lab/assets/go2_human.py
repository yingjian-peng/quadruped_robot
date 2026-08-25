"""Isaac Lab Go2 configuration for rear-leg balance and inverted-pose training."""

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


GO2_HUMAN_CFG = UNITREE_GO2_CFG.copy()
GO2_HUMAN_CFG.init_state.pos = (0.0, 0.0, 0.70)
GO2_HUMAN_CFG.init_state.rot = (0.0, -0.70710678, 0.0, 0.70710678)
GO2_HUMAN_CFG.init_state.joint_pos = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 2.0,
    "RR_thigh_joint": 2.0,
    ".*_calf_joint": -1.7,
}
GO2_HUMAN_CFG.actuators["base_legs"].stiffness = 20.0
GO2_HUMAN_CFG.actuators["base_legs"].damping = 0.5
