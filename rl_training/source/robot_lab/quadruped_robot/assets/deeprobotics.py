# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from quadruped_robot.assets import ROBOT_MODELS_DIR

DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Use the vendor-provided Isaac Sim asset directly.
        usd_path=f"{ROBOT_MODELS_DIR}/deeprobotics/Lite3/Lite3_usd/Lite3.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.8,
            ".*Knee_joint": 1.6,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "Hip": DCMotorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,
            saturation_effort=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=0.5,
            friction=0.0,
        ),
        "Knee": DCMotorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            saturation_effort=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
