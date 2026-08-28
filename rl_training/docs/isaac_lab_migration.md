# Isaac Lab Migration Notes

This directory keeps the Isaac Sim / Isaac Lab migration of the quadruped training project inside the `quadruped_robot` monorepo.

## Current layout

```text
source/robot_lab/
  config/extension.toml
  robot_lab/
    assets/
    tasks/
scripts/reinforcement_learning/rsl_rl/
scripts/tools/
../robot_models/{deeprobotics,unitree}/
```

The legacy Isaac Gym package was removed from this tree.

## Runtime baseline

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

Use the launcher for editable install and execution.

## Asset root

Robot assets are shared from `../robot_models/` and resolved by
`robot_lab.assets.ROBOT_MODELS_DIR`. Deployment keeps its own model interfaces
in `../rl_deploy/` while using the same model assets.

## Task scope

The migrated Isaac Lab tasks cover:

- DeepRobotics Lite3 and M20, flat and rough velocity control
- Unitree A1, B2, Go2, B2W and Go2W, flat and rough velocity control

## Verification order

1. Install the extension with `isaaclab.sh -p -m pip install -e source/robot_lab`.
2. Run `scripts/tools/list_envs.py` and confirm the 14 target task IDs.
3. Run `scripts/tools/smoke_wheel_quadruped.py unitree_go2w --headless --num_envs 1`.
4. Only then start short PPO runs from `scripts/reinforcement_learning/rsl_rl/`.

## Current status

The training subproject keeps the Isaac Lab migration only. Deployment code and shared models are maintained separately in `../rl_deploy/` and `../robot_models/`.
