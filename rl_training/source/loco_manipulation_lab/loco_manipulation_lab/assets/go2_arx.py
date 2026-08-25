"""Isaac Lab articulation configuration for the Go2-ARX robot."""

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from loco_manipulation_lab import ROBOT_MODELS_DIR


GO2_ARX_USD_PATH = (
    ROBOT_MODELS_DIR
    / "go2_arx/urdf/go2_arx/go2_arx.usd"
)


GO2_ARX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(GO2_ARX_USD_PATH),
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
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
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
            stiffness=20.0,
            damping=0.5,
        ),
        "arm": IdealPDActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1000.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
