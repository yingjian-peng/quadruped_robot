# LITE3 Isaac Lab 工程说明

训练项目采用单仓库布局，LITE3 模型由训练与部署代码共享。

## 目录关系

```text
rl_training/source/robot_lab/
  config/extension.toml
  quadruped_robot/
    assets/deeprobotics.py
    tasks/manager_based/locomotion/velocity/
rl_training/scripts/reinforcement_learning/rsl_rl/
rl_training/scripts/tools/
robot_models/deeprobotics/Lite3/
rl_deploy/
```

`quadruped_robot.assets.ROBOT_MODELS_DIR` 从扩展源码位置向上查找仓库根目录，不依赖固定的绝对路径。
Isaac Lab 训练直接加载 `Lite3_usd/Lite3.usd`；部署侧复用同一模型根目录。

## 运行时基线

```text
Isaac Sim: 4.5.0
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

## 任务范围

```text
QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0
QuadrupedRobot-Velocity-Rough-Deeprobotics-Lite3-v0
```

Flat 用于建立基础运动策略；Rough 在同一套观测、动作和机器人参数之上增加程序化地形、高度扫描
和地形难度课程。

## 验证顺序

1. 使用 `isaaclab.sh -p -m pip install --no-build-isolation -e source/robot_lab` 安装扩展。
2. 运行 `scripts/tools/list_envs.py --headless`，确认只有两个 LITE3 任务。
3. 对 Flat 任务运行 `--num_envs 1 --max_iterations 1` 冒烟训练。
4. 训练完整 Flat 基线并执行固定协议 benchmark。
5. 在保持评测协议不变的前提下逐项修改奖励或随机化，每次只验证少量可归因改动。
6. Flat 达标后再进入 Rough 和 Sim-to-Sim，最后根据真机辨识数据推进 Sim-to-Real。
