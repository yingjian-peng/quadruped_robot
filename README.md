# quadruped_robot

面向四足、轮足及移动操作机器人的强化学习工程。仓库采用单仓库（monorepo）结构，将训练、部署和机器人模型分开维护，同时让训练与部署共享同一套模型资产。

## Repository layout

```text
quadruped_robot/
├── rl_training/     # Isaac Sim / Isaac Lab 环境与 RSL-RL PPO 训练
├── rl_deploy/       # Sim-to-Sim、策略导出和 Sim-to-Real 部署
└── robot_models/    # URDF、USD、DAE、STL 等共享机器人模型
```

## Components

- [`rl_training/`](rl_training/README.md)：Isaac Lab 任务、奖励/观测配置、PPO 训练入口和冒烟测试。
- [`rl_deploy/`](rl_deploy/README.md)：部署侧工程骨架，后续承载 Sim-to-Sim 与 Sim-to-Real 实现。
- [`robot_models/`](robot_models/README.md)：训练与部署共享的 DeepRobotics 和 Unitree 模型资产。

## Runtime baseline

当前训练侧参考环境：

```text
Isaac Sim: 4.5.0-rc.36
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

训练环境安装、任务检查和运行方式见 [`rl_training/README.md`](rl_training/README.md)。

## Large assets

机器人网格与 USD 文件通过 Git LFS 管理。克隆或提交本仓库前请安装 Git LFS：

```bash
git lfs install
```

DAE、STL、USD 和 OBJ 文件（包括大小写扩展名）由根目录 `.gitattributes` 统一跟踪。

## Asset licensing

公开分发仓库前，请分别确认机器人模型、网格和第三方资源的再分发许可，并在 `robot_models/README.md` 中记录来源与许可证。代码许可证不自动覆盖第三方模型资产。
