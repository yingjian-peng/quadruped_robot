"""Isaac Lab manager-based B2W-Z1 locomotion-manipulation environment."""

from isaaclab.assets import AssetBaseCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from loco_manipulation_lab.assets import B2W_Z1_CFG

from . import mdp as b2w_z1_mdp


LEG_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
WHEEL_JOINTS = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
ARM_JOINTS = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"]
CONTROLLED_JOINTS = LEG_JOINTS + WHEEL_JOINTS + ARM_JOINTS
CONTROLLED_CFG = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)
GRIPPER_CFG = SceneEntityCfg("robot", body_names="gripperMover")


@configclass
class B2WZ1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = B2W_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 1.0)
        ),
    )
    gripper_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripperMover",
        resampling_time_range=(0.6, 1.2),
        make_quat_unique=True,
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.45), pos_y=(-0.25, 0.25), pos_z=(0.10, 0.55),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    # Action ABI: 12 leg positions, 4 wheel velocities, then 6 arm positions.
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True, preserve_order=True
    )
    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINTS, scale=20.0, use_default_offset=False, preserve_order=True
    )
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.25, use_default_offset=True, preserve_order=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25)
        )
        joint_pos = ObsTerm(
            func=b2w_z1_mdp.joint_pos_rel_without_wheels,
            params={"asset_cfg": CONTROLLED_CFG}, noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": CONTROLLED_CFG}, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05,
        )
        gripper_pos = ObsTerm(func=b2w_z1_mdp.end_effector_position_b, params={"asset_cfg": GRIPPER_CFG})
        gripper_goal = ObsTerm(func=b2w_z1_mdp.pose_command_position, params={"command_name": "gripper_pose"})
        gripper_error = ObsTerm(
            func=b2w_z1_mdp.pose_command_position_delta,
            params={"command_name": "gripper_pose", "asset_cfg": GRIPPER_CFG},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        height_scan = ObsTerm(
            func=b2w_z1_mdp.relative_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5, "scale": 5.0},
        )

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class EventCfg:
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
class RewardsCfg:
    termination = RewTerm(func=mdp.is_terminated, weight=-1.0)
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.5}
    )
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-9)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.2, params={"target_height": 0.5})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 0.5},
    )
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_(thigh|calf)"), "threshold": 0.1},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    object_distance = RewTerm(
        func=b2w_z1_mdp.pose_command_position_exp,
        weight=2.0,
        params={"command_name": "gripper_pose", "asset_cfg": GRIPPER_CFG, "std": 0.316227766},
    )
    object_distance_l2 = RewTerm(
        func=b2w_z1_mdp.pose_command_position_error,
        weight=-1.0,
        params={"command_name": "gripper_pose", "asset_cfg": GRIPPER_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"), "threshold": 1.0},
    )


@configclass
class B2WZ1FlatEnvCfg(ManagerBasedRLEnvCfg):
    seed = 42
    scene: B2WZ1SceneCfg = B2WZ1SceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
