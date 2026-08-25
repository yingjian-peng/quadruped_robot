"""Isaac Lab articulation configurations for the legacy wheel quadrupeds."""

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from loco_manipulation_lab import ROBOT_MODELS_DIR


def _wheel_quadruped_cfg(usd_path: str) -> ArticulationCfg:
    """Build the shared B2W/Go2W articulation configuration.

    The generated USD must not contain active URDF drives.  Isaac Lab owns both
    the leg position and wheel velocity targets through the actuator groups
    below.
    """
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.67,
                ".*_calf_joint": -1.3,
                ".*_foot_joint": 0.0,
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
        },
    )


B2W_USD_PATH = ROBOT_MODELS_DIR / "b2w/usd/b2w.usd"
GO2W_USD_PATH = ROBOT_MODELS_DIR / "go2w/usd/go2w.usd"

B2W_CFG = _wheel_quadruped_cfg(str(B2W_USD_PATH))
GO2W_CFG = _wheel_quadruped_cfg(str(GO2W_USD_PATH))
