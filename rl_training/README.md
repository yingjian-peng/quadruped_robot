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
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p -m pip install --no-build-isolation -e source/robot_lab
```

列出已注册的 Isaac Lab 任务：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

任务 ID 注册于 `quadruped_robot.tasks`，包括：

```text
QuadrupedRobot-Velocity-{Flat,Rough}-Deeprobotics-Lite3-v0
QuadrupedRobot-Velocity-{Flat,Rough}-Unitree-{A1,B2,Go2}-v0
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
  --task QuadrupedRobot-Velocity-Rough-Deeprobotics-Lite3-v0 \
  --headless --num_envs 1 --max_iterations 1
```

短程 PPO 入口位于 `scripts/reinforcement_learning/rsl_rl/` 下。

## 检查点

训练会将检查点写入 `logs/rsl_rl/<experiment_name>/<run>/model_*.pt`。
当前仓库仅包含参数快照（parameter snapshots），未附带已训练的检查点。

## 步态基准评测

不要只通过键盘播放或总 reward 判断策略。`benchmark.py` 会在无 UI、关闭域随机化的
条件下，逐条测试前后、左右平移、原地旋转以及混合指令；每条指令会从相同的初始状态
重复执行，并将每次 rollout 与按类别汇总的指标写成 CSV。核心指标包括三轴速度跟踪误差、
纯旋转时的平移漂移、机身倾角、接触脚世界系滑移速度、四腿占空比、步态时钟匹配度、动作
变化率和关节功率。

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Unitree-Go2-v0 \
  --checkpoint_path logs/rsl_rl/unitree_go2_flat/<run>/model_4999.pt \
  --headless --trials 3 --duration_s 8 --warmup_s 2
```

默认结果写入检查点目录的 `benchmark/rollouts.csv` 和 `benchmark/summary.csv`；可用
`--output_dir` 另行指定目录。横移和转向应优先查看 `lateral`、`yaw` 类别的 `mae_vy`、
`mae_wz`、`yaw_translation_drift_mps`、`stance_slip_mps` 与 `tilt_rad`。

若已有 `rollouts.csv` 但由于中断没有 `summary.csv`，无需重新仿真；可离线生成汇总：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --summarize_only \
  --rollouts_path logs/rsl_rl/unitree_go2_flat/<run>/benchmark/rollouts.csv
```

## Go2 横移/转向微调

当前 Go2 配置会均衡采样前后、横移、原地转和混合速度指令；横移有独立跟踪奖励，固定
对角小跑时钟只对前后主导的指令生效，接触脚滑移使用世界系速度。对于已有的直行策略，
优先从既有 checkpoint 微调，而非从头训练：这会保留已有的站立和前后行走能力。

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Unitree-Go2-v0 \
  --headless --num_envs 4096 --max_iterations 3000 \
  --resume --reset_optimizer --load_run 2026-09-22_17-11-24 --checkpoint model_4999.pt \
  --run_name lateral_yaw_v1
```

`max_iterations=3000` 是在恢复 checkpoint 后追加的 3000 次迭代。完成后，对新 run 的
最终 checkpoint 使用相同的 benchmark 命令，并与旧 checkpoint 的 `summary.csv` 对比。

实测结果（`2026-09-22_17-11-24/model_4999.pt` → `2026-09-23_17-55-23_lateral_yaw_v1/model_7998.pt`）：
三轴速度跟踪误差 −21%~−33%、机身倾角 −34%、接触脚滑移 −14%，但动作变化率 +8%，
步态时钟匹配度 0.673 → 0.575（跌到 stand 基线附近）。原因是 `feet_gait` 权重从 0.5 降到
0.35 并叠加了只对前进主导指令生效的硬门控，步态信号量下降约 62%；同一配置再续训也不再
提升（8000 步时 mean reward 已在 140~145 平台震荡）。

## Go2 步态恢复 v2

v2 只改两处，保持可归因：

1. `feet_gait` 增加 `non_forward_scale`（`mdp/rewards.py`）：侧移/原地转不再被硬门控
   清零，而是按 0.3 的比例打折给时钟奖励；`forward_dominant_only` 仍保留，
   `yaw_to_linear_scale` 0.3 → 0.15（放宽 vx 与 wz 的前进主导判定）。
   `non_forward_scale=0.0` 时行为与旧版完全一致（已用合成数据回归验证）。
2. `joint_mirror` 权重 −0.01 → −0.025（`config/.../unitree_go2/rough_env_cfg.py`）：
   回调左右对称先验，用于压住上一阶段抬头的动作抖动。

续训 1000 次迭代：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Unitree-Go2-v0 \
  --headless --num_envs 4096 --max_iterations 1000 \
  --resume --reset_optimizer \
  --load_run 2026-09-23_17-55-23_lateral_yaw_v1 --checkpoint model_7998.pt \
  --run_name gait_v2
```

评测协议：`--trials 5 --duration_s 8 --warmup_s 2`（比 v1 的 3 trial 更能分辨 5% 级别的差异），
中间 checkpoint（如 `model_8498.pt`）与最后一个 checkpoint 都要测，取指标最优者。

### 验收门槛

门槛定义在 `scripts/tools/compare_benchmarks.py::GATES`，用下面的命令一次性检查所有
已评测 run：

```bash
python3 scripts/tools/compare_benchmarks.py --root logs/rsl_rl/unitree_go2_flat --gates
```

| 指标 | 门槛 | v1（model_7998） |
| --- | --- | --- |
| locomotion `mae_vx` | ≤ 0.08 | 0.061 ✅ |
| locomotion `mae_vy` | ≤ 0.05 | 0.049 ✅ |
| locomotion `mae_wz` | ≤ 0.08 | 0.076 ✅ |
| locomotion `gait_clock_match` | ≥ 0.62 | 0.575 ❌ |
| forward `action_rate_l2` | ≤ 1.25 | 1.438 ❌ |
| stand `actual_wz`（绝对值） | ≤ 0.02 rad/s | 0.068 ❌ |

未列入本阶段但仍待处理：stand 的偏航漂移（可降低 `track_ang_vel_z_exp` 的 `std`，或对零
命令单独加 |ω_z| 惩罚）与 stand 关节功率上升（74.8 → 93.4 W）。
