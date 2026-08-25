"""Manager-based Go2 rear-leg balance and inverted-pose environment."""

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

from loco_manipulation_lab.assets import GO2_HUMAN_CFG

from . import mdp as human_mdp


LEG_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
HIP_CFG = SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"], preserve_order=True)
REAR_HIP_CFG = SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RR_hip_joint"], preserve_order=True)
REAR_FEET_CFG = SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"], preserve_order=True)


@configclass
class Go2RearLegBalanceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = GO2_HUMAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
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
    balance = human_mdp.HumanBalanceCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=human_mdp.HumanBalanceCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), lin_vel_z=(-2.0, 2.0), roll_rate=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True, preserve_order=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        balance_command = ObsTerm(func=human_mdp.human_balance_command, params={"command_name": "balance"}, scale=(2.0, 2.0, 2.0, 0.25))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        height_scan = ObsTerm(
            func=human_mdp.relative_height_scan,
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
    termination = RewTerm(func=mdp.is_terminated, weight=-0.1)
    tracking_lin_vel = RewTerm(func=human_mdp.track_human_lin_vel_exp, weight=2.0, params={"command_name": "balance", "std": 0.5})
    tracking_roll_rate = RewTerm(func=human_mdp.track_human_roll_rate_exp, weight=0.5, params={"command_name": "balance", "std": 0.5})
    lin_vel_x = RewTerm(func=human_mdp.lin_vel_x_l2, weight=-0.5)
    ang_vel_yz = RewTerm(func=human_mdp.angular_velocity_yz_l2, weight=-0.05)
    target_orientation = RewTerm(func=human_mdp.target_projected_gravity_l2, weight=-0.2, params={"target": (1.0, 0.0, 0.0)})
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.42})
    rear_feet_air_time = RewTerm(func=mdp.feet_air_time, weight=1.0, params={"command_name": "balance", "sensor_cfg": REAR_FEET_CFG, "threshold": 0.5})
    rear_single_contact = RewTerm(func=human_mdp.rear_single_contact, weight=0.5, params={"command_name": "balance", "sensor_cfg": REAR_FEET_CFG})
    collision = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_(thigh|calf)"), "threshold": 0.1})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    hip_action = RewTerm(func=human_mdp.hip_joint_abs_sum, weight=-0.5, params={"asset_cfg": HIP_CFG})
    leg_symmetry = RewTerm(func=human_mdp.rear_hip_symmetry_l1, weight=-0.5, params={"asset_cfg": REAR_HIP_CFG})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(base|.*_hip|.*_calf|FL_foot|FR_foot)"), "threshold": 1.0},
    )


@configclass
class Go2RearLegBalanceEnvCfg(ManagerBasedRLEnvCfg):
    seed = 42
    scene: Go2RearLegBalanceSceneCfg = Go2RearLegBalanceSceneCfg(num_envs=4096, env_spacing=3.0)
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
