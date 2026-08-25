# rl_training

This subproject contains the Isaac Sim / Isaac Lab migration for quadruped locomotion and loco-manipulation training.

## What is kept

- `source/loco_manipulation_lab/`
- `scripts/reinforcement_learning/rsl_rl/`
- `scripts/tools/`
- `docs/isaac_lab_migration.md`
- `../robot_models/` (shared with deployment)

The legacy Isaac Gym training package has been removed from this tree.

## Runtime

The current reference setup is:

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 0.48.6
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

Use the Isaac Lab launcher for install and execution:

Run the following commands from the `rl_training/` directory.

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p -m pip install -e source/loco_manipulation_lab
```

List the registered Isaac Lab tasks:

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

Expected task IDs:

```text
Go2-Arx-LocoManip-Flat-v0
Go2-RearLeg-Balance-Flat-v0
B2W-Z1-LocoManip-Flat-v0
B2W-WheelQuadruped-Rough-v0
Go2W-WheelQuadruped-Rough-v0
```

## Asset layout

The Isaac Sim code reads project-owned robot assets from the monorepo-level model directory:

```text
../robot_models/go2_arx/urdf/go2_arx/go2_arx.usd
../robot_models/b2w/urdf/b2w.urdf
../robot_models/go2w/urdf/go2w.urdf
../robot_models/b2w_z1/urdf/b2w_z1.urdf
```

The wheel quadruped converter writes USDs back into the matching `../robot_models/.../usd/` folders.

## Smoke tests

Run the Isaac Lab smoke checks before training:

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/smoke_go2_arx.py --headless --num_envs 1

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/smoke_go2_rear_leg_balance.py --headless --num_envs 1

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/smoke_wheel_quadruped.py b2w --headless --num_envs 1

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/smoke_wheel_quadruped.py go2w --headless --num_envs 1
```

The short PPO entry points are under `scripts/reinforcement_learning/rsl_rl/`.

## Notes

- `Go2-Arx-LocoManip-Flat-v0` is the first migrated Isaac Lab task.
- `Go2-RearLeg-Balance-Flat-v0` keeps the Go2 rear-leg balance behavior in quadruped scope.
- `B2W-Z1-LocoManip-Flat-v0`, `B2W-WheelQuadruped-Rough-v0`, and `Go2W-WheelQuadruped-Rough-v0` are the wheel-legged tasks.
