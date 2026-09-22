# rl_training

面向 DeepRobotics 与 Unitree 运动控制（locomotion）的精简版 Isaac Lab 4.5 训练扩展。
本项目只保留这两类机器人所需的 velocity 任务基类与 RSL-RL 训练配置。
机器人模型从仓库级 `robot_models/` 目录共享。Lite3 训练直接使用供应商提供的
`deeprobotics/Lite3/Lite3_usd/Lite3.usd`，MuJoCo 可视化直接使用同一套资产中的 MJCF。

## 目录结构

```text
source/robot_lab/
  config/extension.toml
  quadruped_robot/
    assets/{deeprobotics,unitree}.py
    tasks/manager_based/locomotion/velocity/
scripts/reinforcement_learning/rsl_rl/
../robot_models/{deeprobotics,unitree}/
```

旧版 loco-manipulation 包与旧版 Isaac Gym 训练包均已从本目录树和环境安装中移除。

## 运行时

当前参考环境配置为：

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

安装与运行均使用 Isaac Lab 启动器。

以下命令请在 `rl_training/` 目录下执行。

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p -m pip install -e source/robot_lab
```

列出已注册的 Isaac Lab 任务：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

任务 ID 注册于 `quadruped_robot.tasks`，包括：

```text
RobotLab-Isaac-Velocity-{Flat,Rough}-Deeprobotics-Lite3-v0
RobotLab-Isaac-Velocity-{Flat,Rough}-Unitree-{A1,B2,Go2}-v0
```

## 资产布局

资产由训练与部署代码共享，并通过
`quadruped_robot.assets.ROBOT_MODELS_DIR` 解析：

```text
../robot_models/deeprobotics/
../robot_models/unitree/
```

## 冒烟测试

训练前先运行 Isaac Lab 冒烟检查：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/list_envs.py --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0 \
  --headless --num_envs 1 --max_iterations 1
```

短程 PPO 入口位于 `scripts/reinforcement_learning/rsl_rl/` 下。

## 检查点

训练会将检查点写入 `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`。
当前仓库仅包含参数快照（parameter snapshots），未附带已训练的检查点。
