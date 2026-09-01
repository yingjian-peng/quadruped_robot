"""Play a trained Isaac Lab RSL-RL checkpoint."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher


sys.path.append(os.path.dirname(__file__))
import cli_args


parser = argparse.ArgumentParser(description="Play an Isaac Lab RSL-RL checkpoint.")
parser.add_argument("--task", required=True, help="Registered Gymnasium task ID.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to display.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Absolute or relative path to a trained .pt policy checkpoint.")
parser.add_argument("--random_velocity_command", action="store_true", help="Use sampled environment velocity commands instead of keyboard control.")
parser.add_argument("--disable_keyboard_control", action="store_true", help="Deprecated alias for --random_velocity_command.")
parser.add_argument("--show_command_arrow", action="store_true", help="Show the base velocity command arrow in the viewport.")
parser.add_argument("--follow_camera", action="store_true", help="Keep the viewport camera following the robot.")
parser.add_argument("--teleop_vx", type=float, default=0.8, help="Keyboard forward/backward velocity command scale.")
parser.add_argument("--teleop_vy", type=float, default=0.4, help="Keyboard lateral velocity command scale.")
parser.add_argument("--teleop_wz", type=float, default=1.0, help="Keyboard yaw velocity command scale.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
cli_args.apply_local_app_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def stabilize_play_reset(env_cfg) -> None:
    """Disable training perturbations that make playback start from unstable poses."""
    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):
        env_cfg.events.randomize_apply_external_force_torque = None
    if hasattr(env_cfg.events, "randomize_push_robot"):
        env_cfg.events.randomize_push_robot = None


def main() -> None:
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.devices import Se2Keyboard
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import quadruped_robot.tasks  # noqa: F401

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.observations.policy.enable_corruption = False
    stabilize_play_reset(env_cfg)
    use_keyboard_control = not (args_cli.random_velocity_command or args_cli.disable_keyboard_control)
    env_cfg.commands.base_velocity.debug_vis = args_cli.show_command_arrow
    if use_keyboard_control:
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    if args_cli.follow_camera:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
    else:
        env_cfg.viewer.origin_type = "world"

    if args_cli.checkpoint_path is not None:
        checkpoint = os.path.abspath(args_cli.checkpoint_path)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    keyboard = None
    if use_keyboard_control:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        keyboard = Se2Keyboard(
            v_x_sensitivity=args_cli.teleop_vx,
            v_y_sensitivity=args_cli.teleop_vy,
            omega_z_sensitivity=args_cli.teleop_wz,
        )
        keyboard._INPUT_KEY_MAPPING.update(
            {
                "W": np.asarray([1.0, 0.0, 0.0]) * args_cli.teleop_vx,
                "S": np.asarray([-1.0, 0.0, 0.0]) * args_cli.teleop_vx,
                "A": np.asarray([0.0, 1.0, 0.0]) * args_cli.teleop_vy,
                "D": np.asarray([0.0, -1.0, 0.0]) * args_cli.teleop_vy,
                "Q": np.asarray([0.0, 0.0, 1.0]) * args_cli.teleop_wz,
                "E": np.asarray([0.0, 0.0, -1.0]) * args_cli.teleop_wz,
            }
        )
        print("[INFO] Keyboard control enabled: hold W/S/A/D/Q/E or arrow/Z/X keys, press L to stop.")

        def apply_keyboard_command() -> None:
            command = torch.as_tensor(keyboard.advance(), device=env.unwrapped.device, dtype=torch.float32)
            command_term.vel_command_b[:] = command

    else:

        def apply_keyboard_command() -> None:
            return None

    dt = env.unwrapped.step_dt
    observations = env.get_observations()
    obs = observations[0] if isinstance(observations, tuple) else observations
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            apply_keyboard_command()
            observations = env.get_observations()
            obs = observations[0] if isinstance(observations, tuple) else observations
            obs, _, _, _ = env.step(policy(obs))
            apply_keyboard_command()
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
