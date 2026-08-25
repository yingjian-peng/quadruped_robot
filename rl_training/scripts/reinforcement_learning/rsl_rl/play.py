"""Play a trained Isaac Lab RSL-RL checkpoint."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


sys.path.append(os.path.dirname(__file__))
import cli_args


parser = argparse.ArgumentParser(description="Play an Isaac Lab RSL-RL checkpoint.")
parser.add_argument("--task", required=True, help="Registered Gymnasium task ID.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to display.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import loco_manipulation_lab.tasks  # noqa: F401

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    observations = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            observations, _, _, _ = env.step(policy(observations))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
