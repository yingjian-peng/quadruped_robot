"""Play or export a trained Isaac Lab RSL-RL checkpoint."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


sys.path.append(os.path.dirname(__file__))
import cli_args


parser = argparse.ArgumentParser(description="Play or export an Isaac Lab RSL-RL checkpoint.")
parser.add_argument("--task", required=True, help="Registered Gymnasium task ID.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a trained .pt checkpoint.")
parser.add_argument(
    "--export_onnx",
    action="store_true",
    help="Export policy.onnx from the checkpoint and exit without opening the UI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.export_onnx:
    args_cli.headless = True
cli_args.apply_local_app_defaults(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def stabilize_play_reset(env_cfg) -> None:
    """Disable training randomization for deterministic playback/export setup."""
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


def configure_keyboard_command(env_cfg) -> None:
    env_cfg.commands.base_velocity.debug_vis = True
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None


def configure_viewer(env_cfg) -> None:
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.viewer.origin_type = "world"


def resolve_checkpoint(agent_cfg) -> str:
    if args_cli.checkpoint_path is not None:
        checkpoint = os.path.abspath(args_cli.checkpoint_path)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")
        return checkpoint

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoints = [
        os.path.abspath(path)
        for root, _, files in os.walk(log_root)
        for path in [os.path.join(root, file) for file in files]
        if os.path.basename(path).startswith("model_") and path.endswith(".pt")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under: {log_root}")
    return max(checkpoints, key=os.path.getmtime)


def export_policy_to_onnx(runner, checkpoint: str) -> str:
    export_model_dir = os.path.join(os.path.dirname(checkpoint), "exported")
    filename = "policy.onnx"
    if hasattr(runner, "export_policy_to_onnx"):
        runner.export_policy_to_onnx(path=export_model_dir, filename=filename)
    else:
        from isaaclab_rl.rsl_rl import export_policy_as_onnx

        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_onnx(policy=policy_nn, normalizer=normalizer, path=export_model_dir, filename=filename)
    return os.path.join(export_model_dir, filename)


def make_keyboard_controller(env):
    import numpy as np
    import torch
    from isaaclab.devices import Se2Keyboard

    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    keyboard = Se2Keyboard(v_x_sensitivity=0.8, v_y_sensitivity=0.4, omega_z_sensitivity=1.0)
    keyboard._INPUT_KEY_MAPPING.update(
        {
            "W": np.asarray([1.0, 0.0, 0.0]) * 0.8,
            "S": np.asarray([-1.0, 0.0, 0.0]) * 0.8,
            "A": np.asarray([0.0, 1.0, 0.0]) * 0.4,
            "D": np.asarray([0.0, -1.0, 0.0]) * 0.4,
            "Q": np.asarray([0.0, 0.0, 1.0]),
            "E": np.asarray([0.0, 0.0, -1.0]),
        }
    )

    def apply_keyboard_command() -> None:
        command = torch.as_tensor(keyboard.advance(), device=env.unwrapped.device, dtype=torch.float32)
        command_term.vel_command_b[:] = command

    return apply_keyboard_command


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import quadruped_robot.tasks  # noqa: F401

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    stabilize_play_reset(env_cfg)
    configure_keyboard_command(env_cfg)
    if args_cli.export_onnx:
        env_cfg.commands.base_velocity.debug_vis = False
    configure_viewer(env_cfg)

    checkpoint = resolve_checkpoint(agent_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)

        if args_cli.export_onnx:
            exported_path = export_policy_to_onnx(runner, checkpoint)
            print(f"[INFO] Exported ONNX policy: {exported_path}")
            return

        policy = runner.get_inference_policy(device=env.unwrapped.device)
        apply_keyboard_command = make_keyboard_controller(env)
        print("[INFO] Keyboard UI mode: use mouse for view, hold W/S/A/D/Q/E or arrow/Z/X keys, press L to stop.")

        obs = env.get_observations()
        obs = obs[0] if isinstance(obs, tuple) else obs
        while simulation_app.is_running():
            with torch.inference_mode():
                apply_keyboard_command()
                obs, _, _, _ = env.step(policy(obs))
                apply_keyboard_command()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
