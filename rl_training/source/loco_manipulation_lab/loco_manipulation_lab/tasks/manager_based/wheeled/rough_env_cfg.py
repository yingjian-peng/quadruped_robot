"""Shared Isaac Lab rough-terrain velocity task for B2W and Go2W."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp as wheel_mdp


LEG_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
WHEEL_JOINTS = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
CONTROLLED_JOINTS = LEG_JOINTS + WHEEL_JOINTS
LEG_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)
CONTROLLED_CFG = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)


@configclass
class WheelQuadrupedSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class WheelQuadrupedCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class WheelQuadrupedActionsCfg:
    # Action ABI: first 12 leg position targets, then four wheel velocity targets.
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True, preserve_order=True
    )
    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINTS, scale=20.0, use_default_offset=False, preserve_order=True
    )


@configclass
class WheelQuadrupedObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25)
        )
        base_height_command = ObsTerm(func=wheel_mdp.base_height_command, params={"target_height": 0.34})
        joint_pos_rel = ObsTerm(
            func=wheel_mdp.joint_pos_rel_without_wheels,
            params={"asset_cfg": CONTROLLED_CFG},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": CONTROLLED_CFG},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        joint_pos = ObsTerm(func=wheel_mdp.joint_pos_without_wheels, params={"asset_cfg": CONTROLLED_CFG})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        height_scan = ObsTerm(
            func=wheel_mdp.relative_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5, "scale": 5.0},
        )

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class WheelQuadrupedEventCfg:
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class WheelQuadrupedRewardsCfg:
    termination = RewTerm(func=mdp.is_terminated, weight=-0.8)
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.632455532}
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.632455532}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0001)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-7)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.34})
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(thigh|calf|base).*"), "threshold": 0.1},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0002)
    stand_still = RewTerm(
        func=wheel_mdp.stand_still_leg_pos_l1,
        weight=-0.01,
        params={"command_name": "base_velocity", "asset_cfg": LEG_CFG},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.9)
    hip_action_l2 = RewTerm(func=wheel_mdp.hip_action_l2, weight=-0.1)


@configclass
class WheelQuadrupedTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class WheelQuadrupedRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration. Concrete robots set ``scene.robot`` in ``__post_init__``."""

    seed = 42
    scene: WheelQuadrupedSceneCfg = WheelQuadrupedSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: WheelQuadrupedObservationsCfg = WheelQuadrupedObservationsCfg()
    actions: WheelQuadrupedActionsCfg = WheelQuadrupedActionsCfg()
    commands: WheelQuadrupedCommandsCfg = WheelQuadrupedCommandsCfg()
    events: WheelQuadrupedEventCfg = WheelQuadrupedEventCfg()
    rewards: WheelQuadrupedRewardsCfg = WheelQuadrupedRewardsCfg()
    terminations: WheelQuadrupedTerminationsCfg = WheelQuadrupedTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
