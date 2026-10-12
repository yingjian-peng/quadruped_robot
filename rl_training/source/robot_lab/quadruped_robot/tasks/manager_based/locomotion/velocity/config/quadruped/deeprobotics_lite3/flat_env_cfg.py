# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

import quadruped_robot.tasks.manager_based.locomotion.velocity.mdp as mdp
from quadruped_robot.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
    TerminationsCfg,
)

##
# 预定义配置
##
from quadruped_robot.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG  # isort: skip




"""DeepRobotics Lite3 平地速度跟踪环境配置。"""
@configclass
class DeeproboticsLite3TerminationsCfg(TerminationsCfg):
    """Lite3 falls terminate; shank contacts remain a reward penalty."""

    fallen_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
    )
    fallen_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class DeeproboticsLite3FlatEnvCfg(LocomotionVelocityEnvCfg):
    terminations: DeeproboticsLite3TerminationsCfg = DeeproboticsLite3TerminationsCfg()
    base_link_name = "TORSO"
    foot_link_name = ".*_FOOT"
    foot_names = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    shank_names = ["FL_SHANK", "FR_SHANK", "HL_SHANK", "HR_SHANK"]
    # fmt: off
    joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        # -----------------------------------------------------------------------------------------------
        # 显示设置（不影响训练，只影响视觉效果）
        # -----------------------------------------------------------------------------------------------
        self.sim.render.rendering_mode = "balanced"         # 渲染预设：performance、balanced、quality
        self.sim.render.antialiasing_mode = "TAA"           # 抗锯齿模式：Off、FXAA、DLSS、TAA、DLAA
        self.sim.render.samples_per_pixel = 4               # 提高每像素采样数，画面更细腻但会降低 FPS

        # ------------------------------场景------------------------------
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.visual_material = sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        )
        self.scene.sky_light.spawn.texture_file = (
            f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
            "kloofendal_43d_clear_puresky_4k.hdr"
        )
        self.scene.sky_light.spawn.intensity = 750.0
        self.commands.base_velocity.debug_vis = False

        # ------------------------------观测------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.6, "command_name": "base_velocity"},
        )
        self.observations.critic.gait_phase = ObsTerm(
            func=mdp.gait_phase,
            params={"period": 0.6, "command_name": "base_velocity"},
        )

        # ------------------------------动作------------------------------
        self.actions.joint_pos.scale = {".*_HipX_joint": 0.125, "^(?!.*_HipX_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------事件------------------------------
        # 平地基准阶段只做小幅重置扰动。稳定后再单独扩大初始状态和动力学随机化。
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [f"^(?!.*{self.base_link_name}).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # 稳定性基准阶段：摩擦接近部署 MJCF 的 1.0，先不加入明显弹跳。
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.95, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.95, 1.0)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.02)
        # 机身标称质量约 5.61 kg；基准阶段限制在约 +/-0.1 kg。
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.1, 0.1)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.98, 1.02)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.005, 0.005),
            "y": (-0.005, 0.005),
            "z": (-0.005, 0.005),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-1.0, 1.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}

        # ------------------------------奖励------------------------------
        # 通用奖励
        self.rewards.is_terminated.weight = 0

        # 机身惩罚
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 0.375 m 是出生根高；策略运行高度目标单独设为 0.32 m。
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height_l2.weight = -20.0
        self.rewards.base_height_l2.params["target_height"] = 0.32
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # E 组只增强低高度保护；阈值保持 0.32 m，宽度项沿用 D 组。
        self.rewards.base_height_below_target.weight = -4.0
        self.rewards.base_height_below_target.params["target_height"] = 0.32
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 关节惩罚
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        # 力矩平方项已抑制过度用力；关闭与其目标高度重叠的功率惩罚。
        self.rewards.joint_power.weight = 0.0
        # 默认角约束稍微放松：仍保留低命令时的 joint_pos_penalty，去掉重复的 stand_still 项。
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 3.0
        # 沿用 nohipx 候选：仅保留对角腿 HipY/Knee 的节拍姿态约束。
        self.rewards.joint_mirror.weight = -0.08
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(HipY|Knee).*", "HR_(HipY|Knee).*"],
            ["FR_(HipY|Knee).*", "HL_(HipY|Knee).*"],
        ]

        # 动作惩罚
        self.rewards.action_rate_l2.weight = -0.04

        # 接触传感器相关奖励
        # 小腿接触不作为跌倒终止，但明确计入接触惩罚；躯干倾倒/过低由独立终止条件处理。
        self.rewards.undesired_contacts.weight = -3.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = self.shank_names
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度跟踪奖励
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_y_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 2.25

        # 其他奖励
        # 步态周期为 0.6 秒、支撑占空比为 56% 时，摆动相约为 0.264 秒
        self.rewards.feet_air_time.weight = 0.3
        self.rewards.feet_air_time.params["threshold"] = 0.22
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 由显式的 trot 时钟规定各足接触相位，关闭泛化的腾空/支撑时间方差项。
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.2
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.5
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.45
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 无改动续训对照：四足使用 nohipx 父策略的统一最小横距。
        self.rewards.feet_stance_width.weight = -1.0
        self.rewards.feet_stance_width.params["min_lateral_distance"] = 0.13
        self.rewards.feet_stance_width.params["asset_cfg"].body_names = self.foot_names
        self.rewards.feet_stance_width.params["asset_cfg"].preserve_order = True
        self.rewards.feet_stance_width.params["sensor_cfg"].body_names = self.foot_names
        self.rewards.feet_stance_width.params["sensor_cfg"].preserve_order = True
        # 以较温和的高度代价保留摆脚离地提示；节拍由 feet_gait 负责。
        self.rewards.feet_height.weight = -2.0
        self.rewards.feet_height.params["target_height"] = 0.08
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.25
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # 前进、横移、混合和原地转向均采用同一对角小跑步态：
        # FL/HR 同步，FR/HL 同步，两组相差半个周期。
        self.rewards.feet_gait.weight = 1.5
        self.rewards.feet_gait.func = mdp.feet_gait
        self.rewards.feet_gait.params = {
            "period": 0.6,
            "offset": [0.0, 0.5, 0.5, 0.0],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_names, preserve_order=True),
            "forward_dominant_only": False,
            "yaw_to_linear_scale": 0.15,
            "non_forward_scale": 1.0,
        }
        # upward 与 flat_orientation_l2 重复；其实现与函数说明不一致，当前禁用该额外姿态奖励。
        self.rewards.upward.weight = 0.0

        # ------------------------------终止条件------------------------------
        # 小腿触地单独惩罚，不复用会把任何小腿触地直接判成失败的 illegal_contact。
        self.terminations.illegal_contact = None

        # ------------------------------课程学习------------------------------
        # 命令范围从最终范围的 20% 开始，并随跟踪能力提升逐步扩大。
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)

        # ------------------------------指令------------------------------
        # 训练与遥操作相同的直接偏航角速度指令。
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        # 独立均匀采样很少产生纯横移或纯转向指令，因此为四种运动模式分配相同的采样比例。
        self.commands.base_velocity.rel_forward_envs = 0.25
        self.commands.base_velocity.rel_lateral_envs = 0.25
        self.commands.base_velocity.rel_yaw_envs = 0.25
        self.commands.base_velocity.rel_mixed_envs = 0.25
        # 恢复 E 组对称采样，使本轮相对 E 组只改变 joint_mirror 奖励。
        self.commands.base_velocity.negative_x_command_probability = 0.5
        self.commands.base_velocity.negative_x_command_probability_mixed = 0.5

        self.disable_zero_weight_rewards()
