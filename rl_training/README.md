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
- 前后、横移、原地转向和混合命令各占 25%，直接训练 yaw-rate 控制；当前纯前后与混合命令的负向 `vx` 比例均为 50%，与 E 组一致。
- 命令课程从完整速度范围的 20% 开始，按当前阈值扩展。
- 对角小跑时钟对前进、横移、混合和原地转向均使用完整权重，固定 `FL/HR` 与 `FR/HL` 两组相差半个周期；周期 `0.6 s`、支撑占空比 `0.56`、步态奖励权重 `1.5`。
- 足端奖励保留时钟接触匹配、腾空时间、接触滑移和温和的摆脚高度提示；关闭重复的足端时间方差和通用接触项。
- 默认关节角和出生根高采用 Lite3 资产值：`HipY=-0.65`、`Knee=1.30` rad，根高 `0.375 m`；当前运行高度奖励目标为 `0.32 m`（权重 `-20`），低于 `0.32 m` 另有保护惩罚（权重 `-4`）。出生高度是 reset 条件，奖励目标是运行时机身高度，两者不需要相同。
- 机身水平惩罚保持 `-5`；支撑脚横向宽度惩罚权重 `-1`，当前恢复到 nohipx 父策略设置：四足最小中心线距离 `0.13 m`。前足阈值 `[0.145,0.145,0.13,0.13] m` 的实验已评测，前足改善但后足收窄，暂不作为主策略。
- 执行器恢复为官网 `DelayedPDActuatorCfg`：Hip/Knee `kp=30`、`kd=1`，力矩限制 24/36 N·m，并带 0–1 个仿真步的延迟；部署 runner 目前仍使用 `kd=0.5` 和 `HipY/Knee=-0.67/1.34`，训练/部署差异需要在新的基准评测后再决定是否统一。
- v3.1 将 `undesired_contacts` 权重设为 `-3`，用于抑制站立时后小腿持续承重；效果需由新训练 checkpoint 评测。
- 复位姿态和域随机化采用适合首阶段训练的中等范围；获得稳定策略后再针对真机辨识结果扩展。
- 站立默认角偏差惩罚仍保留，但站立放大系数从 `5` 降为 `3`；关闭 `stand_still`、额外关节功率代价、足端时间方差和重复的 `upward` 姿态项。
- `joint_mirror=-0.08` 仅约束对角腿 HipY/Knee；沿用已完成评测的 nohipx 候选。
- PPO 学习率为 `5e-4`，网络为 `512 → 256 → 128` 的 ELU MLP。

已完成 `model_2998.pt` GUI基线以及低高度保护 `0.29/0.32 m` 的 A/B 对比。`0.29 m` 组前进误差较低，但7次起停平均下沉 `4.87 cm`；`0.32 m` 组平均下沉 `3.10 cm`。下一步保留 `0.32 m`，对支撑宽度项做单变量消融。完整结果见[优化计划](docs/lite3_optimization_plan_2026-10-11.md)和[迭代记录](docs/lite3_iteration_log.md)。

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --checkpoint_path logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual/model_2998.pt \
  --trials 3 --duration_s 12 --warmup_s 2 \
  --output_dir logs/rsl_rl/deeprobotics_lite3_flat/benchmark_gui_step1_20261011
```

这条命令不带 `--headless`，会显示 Isaac GUI；结果目录会生成 `rollouts.csv`、`summary.csv` 和 `physics_snapshot.json`。另运行一次 forward 起停序列：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --checkpoint_path logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual/model_2998.pt \
  --transition_case forward --trials 1 --warmup_s 0 \
  --transition_stand_s 3 --transition_move_s 8 --transition_recover_s 3 \
  --output_dir logs/rsl_rl/deeprobotics_lite3_flat/benchmark_gui_step1_20261011/transition_forward
```

起停结果会记录开始行走前高度、运动最低高度、停止时高度和站回后的高度。此前 `heightw20_ft250` 的运动根高约 `0.287 m`，综合指标未胜过 `model_2998.pt`。

### 最近一轮微调评测

E组（低高度保护 `-4`、支撑宽度权重 `-1`）是当前步态优先候选：相对D组，它支撑更宽、倾角更小，节拍匹配和滑移相当；高度没有明显提升，速度仍需逐项改善。速度诊断发现后退命令持续欠速，横移时出现前后串动；前进和原地偏航基本跟踪良好。下一轮先针对这些命令方向定位原因，不改步态奖励。完整对照见[迭代记录](docs/lite3_iteration_log.md)。

结果见 [D组固定命令汇总](logs/rsl_rl/deeprobotics_lite3_flat/2026-10-12_11-04-04_lite3_flat_h032_lowguard032_widthw1_ft250/benchmark_gui_review_20261012/summary.csv)、[D组起停测试](logs/rsl_rl/deeprobotics_lite3_flat/2026-10-12_11-04-04_lite3_flat_h032_lowguard032_widthw1_ft250/benchmark_gui_review_20261012/transition_forward/rollouts.csv) 和 [迭代记录](docs/lite3_iteration_log.md)。

### E 组低高度保护权重评测

E 组将低高度保护从 `-2` 加至 `-4` 后，固定命令矩阵仍为33/33无跌倒。相对 D 组，支撑宽度从 `0.2560` 增至 `0.2677 m`、低于 `0.26 m` 的采样比例从 `71.0%` 降至 `36.3%`，倾角P95也降低；但平均高度几乎不变（`0.2936 vs 0.2934 m`），速度误差增大，起步下沉仍约 `3.68 cm`。E 是支撑宽度取向的备选，D 的速度跟踪更好，当前没有全面胜出的版本。

结果见 [E组固定命令汇总](logs/rsl_rl/deeprobotics_lite3_flat/2026-10-12_11-15-12_lite3_flat_h032_lowguardw4_widthw1_ft250/benchmark_gui_review_20261012/summary.csv)、[E组起停测试](logs/rsl_rl/deeprobotics_lite3_flat/2026-10-12_11-15-12_lite3_flat_h032_lowguardw4_widthw1_ft250/benchmark_gui_review_20261012/transition_forward/rollouts.csv) 和 [迭代记录](docs/lite3_iteration_log.md)。

### 纯前后模式 65% 后退样本对照

`model_3745.pt` 与 E 组的 GUI 测试表明：只对纯前后模式提高负向 `vx` 采样，后退跟踪仍有改善，支撑宽度较“纯前后和混合模式均 65%”的候选有所恢复；但横移前向串动、倾角 P95 和整体 `vx` MAE 较 E 组变差。没有跌倒和执行器饱和。当前仍保留 E 组 `model_3496.pt` 作为综合基线，不建议仅凭后退指标选用本候选。详细数据见[迭代记录](docs/lite3_iteration_log.md#2026-10-12-纯前后模式65-后退样本gui复测)。

以下为该实验当时的启动命令；当前源码已切换到下方的 `mirroroff` 配置：

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 250 \
  --resume --load_run 2026-10-12_11-15-12_lite3_flat_h032_lowguardw4_widthw1_ft250 \
  --checkpoint model_3496.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_lowguardw4_widthw1_backcmd65_pure_ft250
```

### 已完成：从 E 组关闭整个关节镜像项

从 E 组 `model_3496.pt` 关闭整个 `joint_mirror` 后，GUI 0.5/0.75 幅值扫描各33回合无跌倒。后退 MAE 从 `0.0892/0.0980` 降到 `0.0632/0.0774 m/s`，但平均支撑宽度由 `0.2640/0.2654` 缩至 `0.2547/0.2584 m`，倾角 P95 由 `0.0351/0.0384` 升至 `0.0439/0.0474 rad`。前进起停的平均下沉由 `3.68 cm` 增至 `4.25 cm`。全关闭方案不满足步态优先目标，保留 E 组作为基线。详细对照见 [迭代记录](docs/lite3_iteration_log.md)。

以下是该实验当时的启动命令；当前源码已切换到下方的前足横距微调配置，不能直接用历史命令复现实验：

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 250 \
  --resume --load_run 2026-10-12_11-15-12_lite3_flat_h032_lowguardw4_widthw1_ft250 \
  --checkpoint model_3496.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_lowguardw4_widthw1_mirroroff_ft250
```

### 已完成：只去掉 HipX 镜像约束

从 E 组 `model_3496.pt` 新开 250 次微调，保留 `joint_mirror=-0.08` 对 HipY/Knee 的对角约束，只移除 HipX 的同号角度比较。负向 `vx` 采样保持 E 组的 50%/50%；高度、足宽、trot 和 PPO 参数不动。训练与 GUI 对照已经完成：后退跟踪和起步下沉改善，但窄支撑比例、擦地率和高速滑移退步。当前仍以 E 组作为步态与足端质量优先的基线；详细数值见 [迭代记录](docs/lite3_iteration_log.md#2026-10-12只去掉-hipx-镜像约束后的-gui-评测)。下方是历史启动命令；当前源码已有前足横距修改，不能直接用该命令复现 nohipx 实验。下一轮命令见后续章节。

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 250 \
  --resume --load_run 2026-10-12_11-15-12_lite3_flat_h032_lowguardw4_widthw1_ft250 \
  --checkpoint model_3496.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_lowguardw4_widthw1_mirror_nohipx_ft250
```

### 已完成：从 nohipx 候选微调前足承重横距

该实验初始化使用 `2026-10-12_13-26-29_lite3_flat_h032_lowguardw4_widthw1_mirror_nohipx_ft250/model_3745.pt`，已完成250次更新。下方保留历史训练命令。前两足的最小横距由 `0.13` 改成 `0.145 m`，后两足保持 `0.13 m`，沿用宽度权重 `-1`。这是机身体坐标系内、单脚到中线的软下限，不是要求脚间距固定为 `0.29 m`；接触掩码和计算坐标系沿用原奖励。左右前足同等处理，目标是减轻前右足内收；高度、相位、摆脚项、命令采样与 PPO 保持父策略配置。

相位诊断发现新增预期摆动相触地主要在换相边缘，候选在 `0.60–0.90` 相位中段未检测到触地，因此本轮不增加摆脚高度代价。E `model_3496.pt` 保留作支撑与步态验收基线。

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 250 \
  --resume --load_run 2026-10-12_13-26-29_lite3_flat_h032_lowguardw4_widthw1_mirror_nohipx_ft250 \
  --checkpoint model_3745.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_nohipx_frontwidth0145_ft250
```

`--max_iterations 250` 表示追加250次更新；`--reset_optimizer` 恢复网络并重置优化器。新 run 的 `params/env.yaml` 中应有 `feet_stance_width.params.min_lateral_distance: [0.145,0.145,0.13,0.13]`（YAML 可能展开成四行列表），weight为 `-1.0`。命令中的 `--headless` 仅用于训练；后续评测继续使用 GUI。

训后按相同的0.5/0.75两档、11类命令×3次及前进起停协议对比父策略与 E：关注前进/前进+逆时针转向的前右足承重横距、后退 MAE、约3.07 cm的父策略起步下沉，以及后退/混合转向的支撑滑移；接触质量用相位分箱区分换相和摆动中段；测试结果已记录在下方评测小结。

本轮 `2026-10-12_15-24-44_.../model_3994.pt` 已完成GUI两档矩阵及起停，共69/69回合无跌倒。前右足在前进时变宽，但左后足整体收窄约13–14 mm，平均支撑宽度降至约0.256 m；后退误差增加，起步下沉从父策略3.07 cm增至3.38 cm。平均滑移与换相接触有所改善，综合上仍不升级为主策略。下一步从同一 nohipx 父检查点做250次阈值不变（四足0.13 m）的续训对照，先排除常规续训漂移。完整数据见[迭代记录](docs/lite3_iteration_log.md#2026-10-12frontwidth0145--model_3994-gui-评测)。

### 下一轮：nohipx 阈值不变的续训对照

当前源码已恢复与父 run 一致的足宽阈值 `0.13 m`、权重 `-1`。从 `model_3745.pt` 新建对照 run，重置优化器并续训250次；这与 frontwidth0145 实验采用相同起点和训练轮数，只取消前足阈值调整。

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 250 \
  --resume --load_run 2026-10-12_13-26-29_lite3_flat_h032_lowguardw4_widthw1_mirror_nohipx_ft250 \
  --checkpoint model_3745.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_nohipx_width013_control_ft250
```

`params/env.yaml` 应记录 `min_lateral_distance: 0.13`。训后用相同 GUI 测评协议与 nohipx `3745`、frontwidth `3994` 和 E `3496` 比较，区分继续训练自然漂移与前足阈值变化的影响。训练命令由用户手动启动。

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

默认仿真条件与 `play.py` 对齐：保留质量、材料、质心和电机增益随机化，并固定初始机身位姿。需要检查名义参数时显式加 `--nominal_conditions`。评测行走命令结束后，脚本会把速度命令清零并继续运行策略，让机器人在零速度目标下保持站立；GUI 默认观察 5 秒，可用 `--hold_window_s 60` 延长。确认可视化结果后，需要批量评测时再显式添加 `--headless`。

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
加 `--phase_diagnostics` 会另存 `phase_diagnostics.csv`：每只脚按本地步态相位分成20段，记录接触率、承重时横向足距、预期摆动相触地率和足端高度。这里的 `swing_scuff_rate` 统计的是**预期摆动相仍有接触**，可能是晚抬脚或提前落脚；只有结合相位段才能判断是否在摆动中途拖地。
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
