# quadruped_robot

`quadruped_robot` 是一个专注于 **DeepRobotics Lite3** 的强化学习运动控制工程。工程以
Isaac Sim / Isaac Lab 为仿真与任务基础，以 RSL-RL PPO 为训练入口，覆盖模型管理、速度跟踪
策略训练、量化评测、Sim-to-Sim 验证以及后续 Sim-to-Real 接入。

![DeepRobotics Lite3](img/lite3.png)

## 一、支持范围

工程只注册一个 Isaac Lab 平地任务：

```text
QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0
```

该任务在本地生成的平面网格上训练基础站立、前后、横移和转向能力，不使用复杂地形、高度扫描或
地形难度课程。策略输出 12 个关节的位置目标增量，底层由 Lite3 的延迟 PD 执行器配置转换为关节力矩。

## 二、参考环境

```text
操作系统: Ubuntu 20.04.6 LTS（内核 5.15.0-139）
GPU: NVIDIA GeForce RTX 5070 Ti
显存: 16 GB（驱动 570.133.20）
Python: 3.10.15（Isaac Sim 自带 Python）
Isaac Sim: 4.5.0
Isaac Lab: 2.3.0（isaaclab 扩展 0.48.6）
PyTorch: 2.7.0+cu128
rsl-rl-lib: 3.1.2
Isaac Lab Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

训练使用 Isaac Sim 自带的 Python，`isaaclab.sh -p` 会自动选择对应解释器。

## 三、工程结构

```text
quadruped_robot/
├── README.md
├── img/lite3.png
├── robot_models/
│   └── deeprobotics/Lite3/
│       ├── Lite3_mjcf/
│       ├── Lite3_urdf/
│       └── Lite3_usd/
├── rl_training/
│   ├── README.md
│   ├── docs/
│   │   ├── isaac_lab_migration.md
│   │   └── lite3_training_strategy.md
│   ├── scripts/
│   │   ├── reinforcement_learning/rsl_rl/
│   │   │   ├── train.py
│   │   │   ├── play.py
│   │   │   └── benchmark.py
│   │   └── tools/
│   │       ├── compare_benchmarks.py
│   │       └── list_envs.py
│   └── source/robot_lab/quadruped_robot/
│       ├── assets/deeprobotics.py
│       └── tasks/manager_based/locomotion/velocity/
└── rl_deploy/
    ├── README.md
    ├── interface/robot/simulation/         # MuJoCo、PyBullet 与模型查看器
    ├── policy/
    ├── run_policy/
    └── state_machine/
```

训练使用 USD，部署仿真使用 URDF/MJCF；机器人模型统一从 `robot_models/deeprobotics/Lite3/` 加载。
`Lite3_description/` 暂不删除，其中的验证场景可继续筛选和使用。

## 四、安装与训练

### 1. 安装与任务检查

以下命令默认从 `rl_training/` 目录执行。先在仓库根目录运行：

```bash
cd rl_training

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p -m pip install \
  --no-build-isolation -e source/robot_lab

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/list_envs.py --headless
```

列表中应只有一个 `QuadrupedRobot-Velocity-*` 任务。

### 2. 训练

Flat 是工程唯一的训练任务。新版 Lite3 默认站姿、延迟 PD 执行器及高度奖励配置从头训练；
当前 v4 训练使用 2048 个并行环境、5000 次迭代：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 5000 \
  --run_name lite3_flat_new_actuator_v4
```

训练输出位于：

```text
logs/rsl_rl/deeprobotics_lite3_flat/<run>/
├── model_*.pt
├── events.out.tfevents.*
└── params/
    ├── env.yaml
    └── agent.yaml
```

### 3. 回放与导出

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/play.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0
```

键盘控制：`W/S` 前后，`A/D` 横移，`Q/E` 转向，`L` 清零停止；也可使用方向键与 `Z/X`。
工具默认加载最新 checkpoint；`--checkpoint_path` 也可以接收具体的 `model_*.pt` 或整个 run 目录。

导出 ONNX：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/play.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --export_onnx
```

导出文件默认写入 checkpoint 同级的 `exported/model_*.onnx`。

### 4. 量化评测

不要只依赖回放观感或总 reward。`benchmark.py` 会重复评测站立、前后、横移、原地转向和混合
指令，并输出速度误差、机身倾角、足端滑移、动作变化率、关节功率、步态时钟匹配度、对角腿
同步率、半周期反相率、摆脚高度和摆动相触地率：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --trials 5 --duration_s 8 --warmup_s 2
```

离线重建最新评测的汇总：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py --summarize_only
```

对比多个已评测 run：

```bash
python3 scripts/tools/compare_benchmarks.py \
  --root logs/rsl_rl/deeprobotics_lite3_flat
```

先用第一条完整 Lite3 基线建立验收门槛，再以相同随机种子、命令矩阵、时长和 trial 数比较配置，
避免把仿真波动误认为改进。训练策略与调参顺序见
[Lite3 训练策略](rl_training/docs/lite3_training_strategy.md)。

## 五、仿真与部署

### 1. Sim-to-Sim

在仓库根目录运行 Lite3 MuJoCo 模型检查：

```bash
python3 rl_deploy/interface/robot/simulation/view_robot.py
python3 rl_deploy/interface/robot/simulation/view_robot.py --gravity
```

`rl_deploy/interface/robot/simulation/` 还包含 MuJoCo 和 PyBullet 仿真接口；
机器人分别加载 `robot_models/` 中的 MJCF 和 URDF。部署程序的编译、仿真运行与控制说明见
[rl_deploy/README.md](rl_deploy/README.md)。模型查看成功不等于训练策略已经完成 Sim-to-Sim 联调。

### 2. Sim-to-Real

本工程尚未完成新策略的真机验证。待 Sim-to-Sim、策略输入输出及安全边界检查通过后，
再接入 Lite3 真机；不将仿真结果直接视为真机运行效果。

## 六、Git LFS

Lite3 的 STL 与 USD 资产通过 Git LFS 管理：

```bash
git lfs install
git lfs ls-files
```

## 七、参考工程

- https://github.com/fan-ziqi/robot_lab
- https://github.com/DeepRoboticsLab/rl_training
- https://github.com/DeepRoboticsLab/Lite3_rl_deploy
