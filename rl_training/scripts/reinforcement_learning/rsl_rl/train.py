"""Train a project task with Isaac Lab 2.x and RSL-RL."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher


sys.path.append(os.path.dirname(__file__))
import cli_args


parser = argparse.ArgumentParser(description="Train an Isaac Lab task with RSL-RL.")
parser.add_argument("--task", required=True, help="Registered Gymnasium task ID.")
parser.add_argument("--num_envs", type=int, default=None, help="Override the configured environment count.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the PPO iteration count.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import quadruped_robot.tasks  # noqa: F401

    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if agent_cfg.resume:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from: {checkpoint}")
        runner.load(checkpoint)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
