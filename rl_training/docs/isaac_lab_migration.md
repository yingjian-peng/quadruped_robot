# Isaac Lab Migration Notes

This directory keeps the Isaac Sim / Isaac Lab migration of the quadruped training project inside the `quadruped_robot` monorepo.

## Current layout

```text
source/loco_manipulation_lab/
  loco_manipulation_lab/
    assets/
    tasks/
scripts/reinforcement_learning/rsl_rl/
scripts/tools/
../robot_models/
```

The legacy Isaac Gym package was removed from this tree.

## Runtime baseline

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 0.48.6
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

Use the launcher for editable install and execution.

## Asset root

Project-owned assets now live in the shared monorepo-level `../robot_models/` directory:

- `../robot_models/go2_arx/urdf/go2_arx/go2_arx.usd`
- `../robot_models/b2w/urdf/b2w.urdf`
- `../robot_models/go2w/urdf/go2w.urdf`
- `../robot_models/b2w_z1/urdf/b2w_z1.urdf`

The wheel-quadruped converter writes the generated USD files back into the matching `usd/` subdirectories in the same tree.

## Task scope

The migrated Isaac Lab tasks are:

- `Go2-Arx-LocoManip-Flat-v0`
- `Go2-RearLeg-Balance-Flat-v0`
- `B2W-Z1-LocoManip-Flat-v0`
- `B2W-WheelQuadruped-Rough-v0`
- `Go2W-WheelQuadruped-Rough-v0`

## Verification order

1. Install the extension with `isaaclab.sh -p -m pip install -e source/loco_manipulation_lab`.
2. Run `scripts/tools/list_envs.py` and confirm the expected task IDs.
3. Run the smoke scripts in `scripts/tools/` with `--headless --num_envs 1`.
4. Only then start short PPO runs from `scripts/reinforcement_learning/rsl_rl/`.

## Current status

The training subproject keeps the Isaac Lab migration only. Deployment code and shared models are maintained separately in `../rl_deploy/` and `../robot_models/`.
