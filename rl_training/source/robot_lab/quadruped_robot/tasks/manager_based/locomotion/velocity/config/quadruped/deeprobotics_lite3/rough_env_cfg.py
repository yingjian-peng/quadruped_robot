# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import quadruped_robot.tasks.manager_based.locomotion.velocity.mdp as mdp
from quadruped_robot.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from quadruped_robot.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG  # isort: skip


@configclass
class DeeproboticsLite3RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "TORSO"
    foot_link_name = ".*_FOOT"
    foot_names = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    # fmt: off
    joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # A deployable phase clock helps the policy settle into a regular gait.
        # It is zeroed for stand commands and uses the same period as feet_gait.
        self.observations.policy.gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.6, "command_name": "base_velocity"},
        )
        self.observations.critic.gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.6, "command_name": "base_velocity"},
        )

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_HipX_joint": 0.125, "^(?!.*_HipX_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        # Start close enough to a recoverable stance that PPO learns locomotion
        # instead of spending most samples on unrecoverable upside-down resets.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (0.0, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.33, 0.33),
                "y": (-0.33, 0.33),
                "z": (-0.33, 0.33),
                "roll": (-0.33, 0.33),
                "pitch": (-0.33, 0.33),
                "yaw": (-0.33, 0.33),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # Use moderate randomization during the first Lite3 training stage. The
        # ranges can be widened after a stable nominal policy has been established.
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.42, 0.88)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.38, 0.72)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.08, 0.42)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.33, 2.33)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.033, 0.033),
            "y": (-0.033, 0.033),
            "z": (-0.033, 0.033),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.75, 1.75)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.75, 1.75)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-6.7, 6.7)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-6.7, 6.7)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.33, 0.33), "y": (-0.33, 0.33)}

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.025
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],
            ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_y_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.75

        # Others
        self.rewards.feet_air_time.weight = 0.3
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.25
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # Full clock guidance for forward-dominant commands and a weaker signal
        # for lateral/yaw commands preserve a regular trot without over-constraining turns.
        self.rewards.feet_gait.weight = 0.45
        self.rewards.feet_gait.func = mdp.feet_gait
        self.rewards.feet_gait.params = {
            "period": 0.6,
            # FL/HR are in phase; FR/HL are in the opposite phase.
            "offset": [0.0, 0.5, 0.5, 0.0],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=self.foot_names, preserve_order=True
            ),
            "forward_dominant_only": True,
            "yaw_to_linear_scale": 0.15,
            "non_forward_scale": 0.3,
        }
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsLite3RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # Begin at 20% of the final command range, then expand as tracking improves.
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)

        # ------------------------------Commands------------------------------
        # Train the same direct yaw-rate command used during teleoperation.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        # Independent uniform sampling rarely produces pure lateral or pure yaw
        # commands, so reserve equal coverage for the four operational modes.
        self.commands.base_velocity.rel_forward_envs = 0.25
        self.commands.base_velocity.rel_lateral_envs = 0.25
        self.commands.base_velocity.rel_yaw_envs = 0.25
        self.commands.base_velocity.rel_mixed_envs = 0.25
