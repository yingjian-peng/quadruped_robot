# quadruped_robot

`quadruped_robot` 是一个面向腿式四足、轮式/轮足四足机器人的强化学习训练与验证工程。工程整合当前主流开源腿式机器人训练框架的常见组织方式，以 Isaac Sim / Isaac Lab 为仿真与任务基础，以 RSL-RL PPO 为主要训练入口，目标是在同一套工程中完成多类型四足机器人模型管理、强化学习任务配置、策略训练、仿真验证和后续部署衔接。

当前版本重点支持 DeepRobotics 和 Unitree 系列机器人，任务以速度跟踪 locomotion 为主，包含平地和粗糙地形两类环境。训练侧 Python 包名统一为 `quadruped_robot`，安装路径为 `rl_training/source/robot_lab`。

## 一、工程环境

本工程当前已在以下本机环境完成基础验证：

```text
操作系统: Ubuntu 22.04.5 LTS
CPU: 13th Gen Intel(R) Core(TM) i5-13490F
内存: 32 GB
GPU: NVIDIA GeForce RTX 2080 Ti x 2
显存: 22 GB x 2
NVIDIA Driver: 580.173.02
CUDA Runtime: PyTorch CUDA 12.1 / 驱动支持 CUDA 13.0
Python: 3.10.20
Conda 环境: pyj_rl_env
Isaac Sim: 4.5.0
Isaac Lab: 2.1.0
PyTorch: 2.5.1+cu121
rsl-rl-lib: 2.3.1
Isaac Lab Launcher: /home/ias/IsaacLab/isaaclab.sh
```

注意：直接运行 `/home/ias/IsaacLab/isaaclab.sh` 前需要先激活 `pyj_rl_env`，否则 launcher 可能找不到当前 Conda 环境中的 Python。

## 二、支持的机器人与任务

当前工程注册了 14 个 Isaac Lab 任务，任务命名格式为：

```text
RobotLab-Isaac-Velocity-{Flat,Rough}-{RobotName}-v0
```

| 类型 | 厂商/系列 | 机器人 | 任务 |
| --- | --- | --- | --- |
| 腿式四足 | DeepRobotics | Lite3 | Flat / Rough velocity tracking |
| 轮式/轮足四足 | DeepRobotics | M20 | Flat / Rough velocity tracking |
| 腿式四足 | Unitree | A1 | Flat / Rough velocity tracking |
| 腿式四足 | Unitree | B2 | Flat / Rough velocity tracking |
| 腿式四足 | Unitree | Go2 | Flat / Rough velocity tracking |
| 轮式/轮足四足 | Unitree | B2W | Flat / Rough velocity tracking |
| 轮式/轮足四足 | Unitree | Go2W | Flat / Rough velocity tracking |

完整任务 ID：

```text
RobotLab-Isaac-Velocity-Flat-Deeprobotics-Lite3-v0
RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0
RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0
RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0
RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0
RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0
RobotLab-Isaac-Velocity-Flat-Unitree-B2-v0
RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0
RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0
RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0
RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0
RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0
RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0
RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0
```

## 三、工程架构

```text
quadruped_robot/
├── README.md
├── rl_training/
│   ├── README.md
│   ├── docs/
│   │   └── isaac_lab_migration.md
│   ├── scripts/
│   │   ├── reinforcement_learning/rsl_rl/
│   │   │   ├── train.py
│   │   │   ├── play.py
│   │   │   └── export_policy_as_jit.py
│   │   └── tools/
│   │       └── list_envs.py
│   └── source/robot_lab/
│       ├── config/extension.toml
│       ├── setup.py
│       └── quadruped_robot/
│           ├── assets/
│           └── tasks/manager_based/locomotion/velocity/
├── rl_deploy/
│   ├── sim_to_sim/
│   └── sim_to_real/
└── robot_models/
    ├── deeprobotics/
    │   ├── lite3_description/
    │   └── m20_description/
    └── unitree/
        ├── a1_description/
        ├── b2_description/
        ├── b2w_description/
        ├── g1_description/
        ├── go2_description/
        └── go2w_description/
```

主要目录说明：

- `rl_training/`：Isaac Lab 训练工程，包含任务注册、环境配置、奖励/观测配置、PPO agent 配置和训练脚本。
- `rl_training/source/robot_lab/quadruped_robot/assets/`：机器人资产入口，统一解析 `robot_models/` 下的模型路径。
- `rl_training/source/robot_lab/quadruped_robot/tasks/`：Isaac Lab Gym 任务注册与 Manager-Based locomotion 任务实现。
- `rl_training/scripts/reinforcement_learning/rsl_rl/`：RSL-RL 训练、回放和策略导出入口。
- `robot_models/`：训练与部署共享的机器人模型资源，包括 URDF、USD、mesh 等。
- `rl_deploy/`：部署侧目录，预留 Sim-to-Sim、Sim-to-Real、策略导出后验证等流程。

## 四、常用命令

以下命令默认从训练目录运行：

```bash
cd /home/ias/pengyingjian_quadrupedrobot/quadruped_robot/rl_training
conda activate pyj_rl_env
```

### 1. 确认当前工程已以 `quadruped_robot` 安装：

```bash
python -m pip show quadruped_robot
```

如果需要重新安装当前工程：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p -m pip install --no-build-isolation -e source/robot_lab
```

### 2. 统计已注册的速度跟踪任务数量，当前应输出 `14`：

```bash
python -c "import gymnasium as gym; import quadruped_robot.tasks; print(len([i for i in gym.registry if i.startswith('RobotLab-Isaac-Velocity-')]))"
```

### 3. 列出 Isaac Lab 可见的任务：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py --headless
```

### 4. 运行最小训练冒烟测试：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --num_envs 1 \
  --max_iterations 1
```

运行粗糙地形冒烟测试：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless \
  --num_envs 1 \
  --max_iterations 1
```

### 5. 正式训练示例：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --num_envs 4096
```

指定 GPU 训练时使用 `--device cuda:0`，不要使用 `CUDA_VISIBLE_DEVICES=0`：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --num_envs 4096 \
  --device cuda:0
```

### 6. 可视化回放训练好的策略：

`play.py` 用于加载训练好的 RSL-RL 策略 checkpoint，并在 Isaac Sim 中观察机器人运动效果。脚本默认使用 `cuda:0`，渲染显示参数沿用 Isaac Lab/Isaac Sim 的默认 GUI 配置。通常只需要运行：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 4
```

默认情况下，`play.py` 会从该任务对应的 `logs/rsl_rl/<experiment_name>/` 中自动加载最新 checkpoint。如果需要指定某一次训练结果：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 4 \
  --load_run 2026-08-29_13-12-12 \
  --checkpoint model_0.pt
```

也可以直接传入模型文件路径：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 4 \
  --checkpoint_path logs/rsl_rl/unitree_go2_flat/2026-08-29_13-12-12/model_0.pt
```

如果需要切换机器人，只替换 `--task`，脚本会按对应任务自动寻找该机器人的最新 checkpoint：

```bash
TERM=xterm /home/ias/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0 \
  --num_envs 10
```

回放时默认启用键盘速度控制，按住 `W/S` 控制前进/后退，`A/D` 控制横向移动，`Q/E` 控制转向，`L` 清零停止。也可以使用方向键和 `Z/X`。如果需要使用环境随机采样的速度命令，可添加 `--disable_keyboard_control`。

黑屏原因说明：本机双 GPU 同时被 Isaac Sim 激活时，viewport 渲染路径可能不稳定。本工程默认固定训练和回放使用 `cuda:0`，并在 Kit 启动参数中关闭 multi-GPU、指定渲染 GPU0 和物理 GPU0；其它显示质量、分辨率、DLSS 等参数保持 Isaac Lab/Isaac Sim 默认配置，便于和 `train.py` 的显示效果保持一致。

### 7. 查看训练日志和模型：

```bash
ls logs/rsl_rl
```

训练输出默认写入：

```text
rl_training/logs/rsl_rl/<experiment_name>/<timestamp>/
```

其中包含 `model_*.pt`、`params/env.yaml`、`params/agent.yaml` 和 TensorBoard event 文件。

## 五、Git LFS

机器人网格与 USD 文件通过 Git LFS 管理。克隆或提交本仓库前请安装并初始化 Git LFS：

```bash
git lfs install
```

DAE、STL、USD 和 OBJ 文件由根目录 `.gitattributes` 统一跟踪。




## 参考工程

- （fan-ziqi）https://github.com/fan-ziqi/robot_lab
- （DeepRoboticsLab）https://github.com/DeepRoboticsLab/rl_training
