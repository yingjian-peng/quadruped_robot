# rl_training

Focused Isaac Lab 4.5 training extension for DeepRobotics and Unitree locomotion.
The project keeps only the velocity task base and RSL-RL training
configurations required by these two robot families. Robot models are shared
from the repository-level `robot_models/` directory.

## Layout

```text
source/robot_lab/
  config/extension.toml
  quadruped_robot/
    assets/{deeprobotics,unitree}.py
    tasks/manager_based/locomotion/velocity/
scripts/reinforcement_learning/rsl_rl/
../robot_models/{deeprobotics,unitree}/
```

The old loco-manipulation package and its ARX/Z1 task registrations are not part
of this training project.

The legacy Isaac Gym training package has been removed from this tree.

## Runtime

The current reference setup is:

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

Use the Isaac Lab launcher for install and execution:

Run the following commands from the `rl_training/` directory.

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p -m pip install -e source/robot_lab
```

List the registered Isaac Lab tasks:

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

The task IDs are registered from `quadruped_robot.tasks` and include:

```text
RobotLab-Isaac-Velocity-{Flat,Rough}-Deeprobotics-Lite3-v0
RobotLab-Isaac-Velocity-{Flat,Rough}-Deeprobotics-M20-v0
RobotLab-Isaac-Velocity-{Flat,Rough}-Unitree-{A1,B2,Go2}-v0
RobotLab-Isaac-Velocity-{Flat,Rough}-Unitree-{B2W,Go2W}-v0
```

## Asset layout

Assets are shared by training and deployment code and are resolved through
`quadruped_robot.assets.ROBOT_MODELS_DIR`:

```text
../robot_models/deeprobotics/
../robot_models/unitree/
```

## Smoke tests

Run the Isaac Lab smoke checks before training:

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/list_envs.py --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless --num_envs 1 --max_iterations 1
```

The short PPO entry points are under `scripts/reinforcement_learning/rsl_rl/`.

## Checkpoints

Training writes checkpoints to `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`.
The repository currently contains parameter snapshots only; no trained checkpoint
is included.
