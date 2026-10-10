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
| 官网 Lite3 USD、默认姿态与电机参数 | `quadruped_robot/assets/deeprobotics.py` |
| 通用 Scene、观测、动作、事件和奖励模板 | `velocity/velocity_env_cfg.py` |
| LITE3 平地、奖励、命令、域随机化和足端映射 | `deeprobotics_lite3/flat_env_cfg.py` |
| PPO 网络与优化器参数 | `deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py` |

当前 Lite3 资产、观测、动作、域随机化、奖励和终止配置逐项说明见
[Lite3 平地训练配置](docs/lite3_training_config.md)。

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

## 历史 v3.1 续训记录（旧配置）

下面命令和结果记录的是旧 v3.1 实验，不代表当前源码配置，也不建议把该 checkpoint 当作本次奖励修改后的策略。历史 checkpoint 的准确配置应以 run 目录中的 `params/env.yaml` 和 `params/agent.yaml` 为准。当前任务仍是 Flat，新的训练应使用清楚的新 `run_name`，并按本页“当前训练策略”和配置文档选择从头训练或兼容 checkpoint 续训。

旧实验当时使用的命令：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 300 \
  --resume --load_run 2026-10-05_23-26-13_lite3_flat_posture_v3 \
  --checkpoint model_4999.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_shank_penalty_v3_1_ft300
```

`--max_iterations 300` 是从 checkpoint **再训练 300 次**，不是总共训练到第 300 次。新 run 单独保存，原 v3 checkpoint 不会被覆盖。`--reset_optimizer` 重新初始化 PPO 优化器，但保留策略权重和迭代号；`--resume_full_command_range` 让新环境从完整速度命令范围开始，避免命令课程在续训时退回 20%。先比较续训 checkpoint 与 v3 基线的后足/后小腿接触、机身高度、速度误差和跌倒率；若仍有改善空间，再从该 run 追加训练。

这轮 300 次续训已完成，run 为 `2026-10-09_18-48-24_lite3_flat_shank_penalty_v3_1_ft300/`。固定协议评测表明站立偏航和部分速度命令改善，但右后足仍不着地、右后小腿持续承重，运动机身高度从 v3 的 `0.299 m` 降到 `0.294 m`。该候选未解决主要姿态问题；这些是旧资产/执行器配置下的历史结果，不代表当前资产和奖励配置的效果。完整对比见 [v3 迭代记录](docs/lite3_iteration_log.md)。

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
- 命令课程从完整速度范围的 20% 开始，按当前阈值扩展。
- 对角小跑时钟对前进、横移、混合和原地转向均使用完整权重，固定 `FL/HR` 与 `FR/HL` 两组相差半个周期；周期 `0.6 s`、支撑占空比 `0.56`、步态奖励权重 `1.5`。
- 足端奖励保留时钟接触匹配、腾空时间、接触滑移和温和的摆脚高度提示；关闭重复的足端时间方差和通用接触项。
- 默认关节角和出生根高采用 Lite3 资产值：`HipY=-0.65`、`Knee=1.30` rad，根高 `0.375 m`；运行高度奖励目标为 `0.32 m`（权重 `-12`），低于 `0.29 m` 另有轻量保护惩罚。出生高度是 reset 条件，奖励目标是运行时机身高度，两者不需要相同。
- 机身水平惩罚保持 `-5`；支撑脚新增最小横向宽度惩罚（权重 `-2`、最小中心线距离 `0.13 m`），只用于抑制承重腿内收，不约束摆动腿轨迹。
- 执行器恢复为官网 `DelayedPDActuatorCfg`：Hip/Knee `kp=30`、`kd=1`，力矩限制 24/36 N·m，并带 0–1 个仿真步的延迟；部署 runner 目前仍使用 `kd=0.5` 和 `HipY/Knee=-0.67/1.34`，训练/部署差异需要在新的基准评测后再决定是否统一。
- v3.1 将 `undesired_contacts` 权重设为 `-3`，用于抑制站立时后小腿持续承重；效果需由新训练 checkpoint 评测。
- 复位姿态和域随机化采用适合首阶段训练的中等范围；获得稳定策略后再针对真机辨识结果扩展。
- 站立默认角偏差惩罚仍保留，但站立放大系数从 `5` 降为 `3`；关闭 `stand_still`、额外关节功率代价、足端时间方差和重复的 `upward` 姿态项。
- PPO 学习率为 `5e-4`，网络为 `512 → 256 → 128` 的 ELU MLP。

针对 `model_2998.pt` 的下一步建议小步续训，不从头训练：步态、速度跟踪和机身姿态已经基本成形，本轮只改奖励目标，观测和动作接口没变。先训练 300 次，再用同一套固定命令 GUI 评测，确认高度改善且步态没有退化。

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 300 \
  --resume \
  --load_run 2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual \
  --checkpoint model_2998.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_stancewidth_ft300
```

这里的 `0.32 m` 是运行高度奖励目标；`0.375 m` 仍是出生根高。默认关节角的静态接地根高约在 `0.344 m` 附近，把出生根高直接降到 `0.32 m` 可能让碰撞几何初始穿地，反而加重接触冲击。

逐项配置见 [docs/lite3_training_config.md](docs/lite3_training_config.md)；调参背景和顺序见 [docs/lite3_training_strategy.md](docs/lite3_training_strategy.md)。

此前 v3/v3.1 的评测记录对应旧执行器和碰撞层配置，只作为历史排查资料。用户已确认先前零动作测试中的翻倒与初始接触冲击有关，因此不能用它单独判断 PD 支撑能力；当前策略以 `model_2998.pt` 的固定命令表现作为迭代基线。历史数据与更正记录见 [v3 迭代记录](docs/lite3_iteration_log.md)。

## 官网 Lite3 参考对照

`/home/robot/pengyingjian_external/rl_training` 使用 Isaac Sim 5.1.0 / Isaac Lab 2.3.2；本工程运行环境是 Isaac Sim 4.5.0 / Isaac Lab 2.3.0。Lite3 USD 和 URDF/MJCF 网格资产相同，但官网 USD 基础层此前被本工程改过小腿碰撞体，现已恢复为下载包原文件。动作缩放与官网 Lite3 配置一致；本工程保留独立的平地奖励、窄域随机化、跌倒终止和 2 维步态相位输入。由于通用环境、奖励项和训练入口代码版本差异较大，当前只迁移 Lite3 模型/执行器基线，不整体覆盖官网训练框架。

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

`benchmark.py` 默认打开 Isaac Sim 图形界面，不传 `--headless` 时可以直接观察评测。默认仿真条件与 `play.py` 对齐，并行执行固定命令矩阵。要单独观察某个命令，传 `--case` 并将 `--trials` 设为 `1`，窗口会跟随唯一评测环境：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --checkpoint_path logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual/model_2998.pt \
  --case forward --trials 1 --duration_s 8 --warmup_s 2
```

默认仿真条件与 `play.py` 对齐：保留质量、材料、质心和电机增益随机化，并固定初始机身位姿。需要检查名义参数时显式加 `--nominal_conditions`。评测行走命令结束后，脚本会把速度命令清零并继续运行策略，让机器人在零速度目标下保持站立；GUI 默认观察 20 秒，可用 `--hold_window_s 60` 延长。确认可视化结果后，需要批量评测时再显式添加 `--headless`。

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

`rollouts.csv` 还记录四足和四条小腿的接触占空比（`duty_FL_FOOT`、`duty_HL_SHANK` 等），以及首次跌倒的步数、时间、机身高度和倾角、`fall_by_height`、
`fall_by_tilt`，用于区分两种判据。另有 `roll_abs_rad`、`pitch_abs_rad` 和 HipX/HipY/Knee 关节偏差均值，便于分别检查机身横滚、俯仰及步态中各关节偏离默认姿态的程度。`measured_duration_s=0` 只表示预热期后没有有效样本。
排查默认站姿时可加 `--zero_actions`，在同一环境中输出零关节增量；该结果不代表策略性能。`--trace_pose` 可打印复位关节角、足端高度、接触、动作及力矩；`--initial_height_m`、`--initial_hip_y_rad`、`--initial_knee_rad`、`--actuator_damping`、`--actuator_stiffness`、`--solver_velocity_iterations`、`--solver_position_iterations` 可用于位姿、执行器与物理求解器诊断。`--linear_command_scale`、`--yaw_command_scale` 可把固定命令缩至训练课程内，结果需用独立 `--output_dir` 保存。

跌倒统计会同时读取环境自动复位前的 Termination Manager 条件，避免把复位后的机器人姿态误当成未跌倒。
2026-10-10 的官网参数对照结果及当前限制见 [Lite3 迭代记录](docs/lite3_iteration_log.md)。

比较所有已评测 run：

```bash
python3 scripts/tools/compare_benchmarks.py \
  --root logs/rsl_rl/deeprobotics_lite3_flat
```

以 v3 的完整固定协议评测作为后续迭代基线。应依据 LITE3 的速度范围、真机安全
余量和多次仿真方差，确定 `mae_vx/vy/wz`、`tilt_rad`、`stance_slip_mps`、`action_rate_l2`、
`joint_deviation_mean_abs_rad`、
`joint_power_w`、`gait_clock_match`、`diag_sync_fl_hr/fr_hl`、`diag_antiphase`、
`swing_height_m` 与 `swing_scuff_rate` 的验收区间。
