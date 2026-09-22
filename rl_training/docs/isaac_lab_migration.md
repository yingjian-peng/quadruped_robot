# Isaac Lab 迁移说明

本目录将四足机器人训练项目的 Isaac Sim / Isaac Lab 迁移内容保留在 `quadruped_robot` 单体仓库（monorepo）内。

## 当前目录结构

```text
source/robot_lab/
  config/extension.toml
  quadruped_robot/
    assets/
    tasks/
scripts/reinforcement_learning/rsl_rl/
scripts/tools/
../robot_models/{deeprobotics,unitree}/
```

旧版 Isaac Gym 包已从本目录树中移除。

## 运行时基线

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

可编辑安装（editable install）与运行均使用该启动器。

## 资产根目录

机器人资产从 `../robot_models/` 共享，并由
`quadruped_robot.assets.ROBOT_MODELS_DIR` 解析。部署代码在 `../rl_deploy/`
中保留自己的模型接口，同时复用同一套模型资产。

## 任务范围

迁移后的 Isaac Lab 任务涵盖：

- DeepRobotics Lite3，平地与崎岖地形的速度控制
- Unitree A1、B2 与 Go2，平地与崎岖地形的速度控制

## 验证顺序

1. 使用 `isaaclab.sh -p -m pip install -e source/robot_lab` 安装扩展。
2. 运行 `scripts/tools/list_envs.py`，确认 8 个目标任务 ID。
3. 运行 `scripts/reinforcement_learning/rsl_rl/train.py --task <TASK_ID> --headless --num_envs 1 --max_iterations 1`，确认任务能创建并完成一次迭代。
4. 确认无误后再按目标环境数启动完整 PPO 训练。

## 当前状态

训练子项目仅保留 Isaac Lab 迁移内容。部署代码与共享模型分别在 `../rl_deploy/` 与 `../robot_models/` 中独立维护。
