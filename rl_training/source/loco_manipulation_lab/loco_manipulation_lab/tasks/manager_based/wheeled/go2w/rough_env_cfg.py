"""Go2W wheel-quadruped velocity configuration."""

from isaaclab.utils import configclass

from loco_manipulation_lab.assets import GO2W_CFG

from ..rough_env_cfg import WheelQuadrupedRoughEnvCfg


@configclass
class Go2WRoughEnvCfg(WheelQuadrupedRoughEnvCfg):
    """Migrates Go2W's forward/backward-only rough-terrain policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"
        self.scene.num_envs = 6000
        self.scene.terrain.max_init_terrain_level = 5
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.collision.params["sensor_cfg"].body_names = ".*(thigh|calf|base).*"
