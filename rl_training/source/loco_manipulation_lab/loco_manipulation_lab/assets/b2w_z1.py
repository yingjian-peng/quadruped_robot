"""Isaac Lab articulation configuration for the B2W-Z1 wheel-legged quadruped."""

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from loco_manipulation_lab import ROBOT_MODELS_DIR


B2W_Z1_USD_PATH = ROBOT_MODELS_DIR / "b2w_z1/usd/b2w_z1.usd"

B2W_Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(B2W_Z1_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.60),
        joint_pos={
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
            "arm_joint[1-6]": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=["(FL|FR|RL|RR)_(hip|thigh|calf)_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=50.0,
            damping=1.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["(FL|FR|RL|RR)_foot_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.5,
        ),
        "arm": IdealPDActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
