# rl_training

面向 DeepRobotics LITE3 速度控制的 Isaac Lab 4.5 训练扩展。训练目标包括稳定站立、前后移动、
横移、原地转向、混合速度跟踪，以及在程序化崎岖地形上的适应能力。

## 目录与配置入口

```text
source/robot_lab/
  config/extension.toml
  quadruped_robot/
    assets/deeprobotics.py
    tasks/manager_based/locomotion/velocity/
      velocity_env_cfg.py
      mdp/
      config/quadruped/deeprobotics_lite3/
scripts/
  reinforcement_learning/rsl_rl/{train,play,benchmark}.py
  tools/{list_envs,compare_benchmarks}.py
../robot_models/deeprobotics/Lite3/
```

关键配置：

| 内容 | 文件 |
| --- | --- |
| LITE3 USD、默认姿态与电机参数 | `quadruped_robot/assets/deeprobotics.py` |
| 通用 Scene、观测、动作、事件和奖励模板 | `velocity/velocity_env_cfg.py` |
| LITE3 奖励、命令、域随机化和足端映射 | `deeprobotics_lite3/rough_env_cfg.py` |
| Flat 对 Rough 的覆盖项 | `deeprobotics_lite3/flat_env_cfg.py` |
| PPO 网络与优化器参数 | `deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py` |

## 运行环境与安装

参考运行时：

```text
Isaac Sim: 4.5.0
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

以下命令均在 `rl_training/` 目录下执行：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p -m pip install \
  --no-build-isolation -e source/robot_lab
```

列出已注册任务：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/tools/list_envs.py --headless
```

预期任务：

```text
QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0
QuadrupedRobot-Velocity-Rough-Deeprobotics-Lite3-v0
```

## 冒烟测试

先确认任务可以创建并完成一次 PPO 迭代：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 1 --max_iterations 1
```

## 正式训练

建议先训练 Flat 基线：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 4096 --run_name baseline_v1
```

Flat 策略稳定后再训练 Rough：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Rough-Deeprobotics-Lite3-v0 \
  --headless --num_envs 4096 --run_name rough_v1
```

检查点写入：

```text
logs/rsl_rl/deeprobotics_lite3_{flat|rough}/<run>/model_*.pt
```

每个 run 同时保存 TensorBoard event、`params/env.yaml` 和 `params/agent.yaml`。比较实验时应保留这些
参数快照，不能只保存 checkpoint。

## 当前训练策略

LITE3 配置包含以下机制：

- actor 使用可部署的本体观测、速度命令、上一帧动作和 0.6 s 步态相位；不依赖 base 线速度或地形高度图。
- critic 在 Rough 任务中额外使用 base 线速度和高度扫描，形成非对称 actor-critic。
- 前进、横移、原地转向和混合命令各占 25%，直接训练 yaw-rate 控制。
- 命令课程从完整速度范围的 20% 开始，跟踪能力达标后逐步扩展到 100%。
- 对角小跑时钟对前进指令使用完整权重，对横移和转向只保留较弱约束。
- 足端奖励同时约束腾空时间、时序一致性、接触滑移和相对机身的摆脚高度。
- 复位姿态和域随机化采用适合首阶段训练的中等范围；获得稳定策略后再针对真机辨识结果扩展。
- PPO 学习率为 `5e-4`，网络为 `512 → 256 → 128` 的 ELU MLP。

详细参数和调参顺序见 [docs/lite3_training_strategy.md](docs/lite3_training_strategy.md)。

## 回放与导出

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/play.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --checkpoint_path logs/rsl_rl/deeprobotics_lite3_flat/<run>/model_<iteration>.pt
```

加上 `--export_onnx` 可将策略导出到 checkpoint 同级的 `exported/policy.onnx`。

## 固定协议评测

`benchmark.py` 会关闭观测噪声和域随机化，并行执行固定命令矩阵。每个候选 checkpoint 使用相同
参数评测：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --checkpoint_path logs/rsl_rl/deeprobotics_lite3_flat/<run>/model_<iteration>.pt \
  --headless --trials 5 --duration_s 8 --warmup_s 2
```

结果写入 checkpoint 目录下的 `benchmark/rollouts.csv` 和 `benchmark/summary.csv`。若仿真在生成
汇总前中断，可离线恢复：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --summarize_only \
  --rollouts_path logs/rsl_rl/deeprobotics_lite3_flat/<run>/benchmark/rollouts.csv
```

比较所有已评测 run：

```bash
python3 scripts/tools/compare_benchmarks.py \
  --root logs/rsl_rl/deeprobotics_lite3_flat
```

首条完整基线产生前不预设其他机器人的数值门槛。基线完成后，应依据 LITE3 的速度范围、真机安全
余量和多次仿真方差，确定 `mae_vx/vy/wz`、`tilt_rad`、`stance_slip_mps`、`action_rate_l2`、
`joint_power_w` 与 `gait_clock_match` 的验收区间。
