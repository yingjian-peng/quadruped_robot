# rl_training

面向 DeepRobotics LITE3 平地速度控制的 Isaac Lab 4.5 训练扩展。训练目标包括稳定站立、前后
移动、横移、原地转向和混合速度跟踪。

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
| LITE3 平地、奖励、命令、域随机化和足端映射 | `deeprobotics_lite3/flat_env_cfg.py` |
| PPO 网络与优化器参数 | `deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py` |

## 运行环境与安装

参考运行时：

```text
Isaac Sim: 4.5.0
Isaac Lab: 2.3.0
Launcher: /home/robot/isaacsim/IsaacLab/isaaclab.sh
```

以下命令均在 `rl_training/` 目录下执行。新开终端后先运行：

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
```

安装扩展：

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

Flat 是唯一训练任务。新站姿和奖励配置需从头训练，建议先按 1048 个环境运行 5000 次迭代：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 5000 \
  --run_name lite3_flat_posture_v3
```

检查点写入：

```text
logs/rsl_rl/deeprobotics_lite3_flat/<run>/model_*.pt
```

每个 run 同时保存 TensorBoard event、`params/env.yaml` 和 `params/agent.yaml`。比较实验时应保留这些
参数快照，不能只保存 checkpoint。

## 当前训练策略

LITE3 配置包含以下机制：

- actor 使用可部署的本体观测、速度命令、上一帧动作和 0.6 s 步态相位；不依赖 base 线速度或地形高度图。
- critic 使用仿真可直接获得的 base 线速度，但不使用地形高度扫描。
- 前进、横移、原地转向和混合命令各占 25%，直接训练 yaw-rate 控制。
- 命令课程从完整速度范围的 20% 开始，跟踪能力达标后逐步扩展到 100%。
- 对角小跑时钟对前进、横移、混合和原地转向均使用完整权重，固定 `FL/HR` 与 `FR/HL` 两组相差半个周期。
- 足端奖励同时约束腾空时间、时序一致性、接触滑移和摆脚的世界坐标高度。
- 默认关节姿态对应约 0.34 m 几何站高；高度奖励重点压制 0.32 m 以下的下蹲，同时限制腿部偏离默认姿态和动作突变。
- 复位姿态和域随机化采用适合首阶段训练的中等范围；获得稳定策略后再针对真机辨识结果扩展。
- PPO 学习率为 `5e-4`，网络为 `512 → 256 → 128` 的 ELU MLP。

详细参数和调参顺序见 [docs/lite3_training_strategy.md](docs/lite3_training_strategy.md)。

## 回放与导出

训练和 PLAY 默认启用参考工程已验证的 `balanced + TAA + 4 samples_per_pixel` 渲染设置，以便观察足端接触、机身姿态和动作细节。
它不会改变仿真步长或 50 Hz 控制周期；需要降低可视化开销时可显式传入
`--rendering_mode balanced` 或 `--rendering_mode performance`。

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/play.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0
```

不传 `--checkpoint_path` 时会自动使用最新 run 中迭代编号最大的 checkpoint；也可以给
`--checkpoint_path` 传入一个具体的 `model_*.pt` 文件或整个 run 目录。相对路径从任意目录调用
都能解析。加上 `--export_onnx` 可将策略导出到 checkpoint 同级的 `exported/model_*.onnx`。

## 固定协议评测

`benchmark.py` 会关闭观测噪声和域随机化，并行执行固定命令矩阵。每个候选 checkpoint 使用相同
参数评测：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --trials 5 --duration_s 8 --warmup_s 2
```

它同样默认评测最新 checkpoint。结果写入 checkpoint 所在 run 的 `benchmark/rollouts.csv` 和
`benchmark/summary.csv`。若仿真在生成汇总前中断，可对最新一份 `rollouts.csv` 离线恢复：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --summarize_only
```

需要指定旧结果时再传 `--rollouts_path`。评测默认以机身高度低于 0.18 m 或倾角超过 0.8 rad
判定跌倒（可用 `--fall_height_m`、`--fall_tilt_rad` 调整），`fall_detected` 列记录结果；
跌倒或环境终止后不再把后续状态混入均值。站立指令的步态时钟、对角腿同步和摆脚指标记为 `NaN`，
这些指标只用于运动指令。

比较所有已评测 run：

```bash
python3 scripts/tools/compare_benchmarks.py \
  --root logs/rsl_rl/deeprobotics_lite3_flat
```

首条完整基线产生前不预设其他机器人的数值门槛。基线完成后，应依据 LITE3 的速度范围、真机安全
余量和多次仿真方差，确定 `mae_vx/vy/wz`、`tilt_rad`、`stance_slip_mps`、`action_rate_l2`、
`joint_deviation_mean_abs_rad`、
`joint_power_w`、`gait_clock_match`、`diag_sync_fl_hr/fr_hl`、`diag_antiphase`、
`swing_height_m` 与 `swing_scuff_rate` 的验收区间。
