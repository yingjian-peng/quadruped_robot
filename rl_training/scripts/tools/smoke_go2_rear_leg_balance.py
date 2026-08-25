"""Create the Go2 rear-leg balance environment and execute one zero-action step."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the Go2 rear-leg balance Isaac Lab smoke test.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import loco_manipulation_lab.tasks  # noqa: F401

    task_id = "Go2-RearLeg-Balance-Flat-v0"
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = gym.make(task_id, cfg=env_cfg)
    observations, _ = env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    policy_dim = observations["policy"].shape[-1]
    critic_dim = observations["critic"].shape[-1]
    env.step(torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device))
    print(f"SMOKE_OK task={task_id} actions={action_dim} policy_obs={policy_dim} critic_obs={critic_dim}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
