# Lite3 运动控制迭代记录

本文整理截至 2026-10-08 的本轮对话，供后续新对话继续迭代。**用户观察、已有 benchmark 数据和待验证假设分开记录**；训练启动或保存 checkpoint 不等于策略质量已经达标。本文是历史记录，当前生效配置以代码和对应 run 的 `params/env.yaml`、`params/agent.yaml` 为准。

## 1. 范围与配置入口

- 只维护 DeepRobotics Lite3 的平地速度任务：`QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0`。复杂地形和其他机器狗不在当前训练范围；早期 Rough 日志不迁移到 Flat。Go2 曾作为步态效果参考，但不是当前训练目标。
- 机器人 USD、默认关节角及执行器：[资产配置](../source/robot_lab/quadruped_robot/assets/deeprobotics.py)。训练任务的观测、命令、奖励、域随机化和渲染：[Flat 配置](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_env_cfg.py)。
- 训练、回放和固定协议评测分别使用 `rl_training/scripts/reinforcement_learning/rsl_rl/{train,play,benchmark}.py`。所有命令默认在 `rl_training/` 目录运行。
- `robot_models/deeprobotics/Lite3/` 统一保存 USD、URDF 和 MJCF；`rl_deploy/` 已引入仿真和部署代码，但本项目训练策略尚未完成 Sim-to-Sim 联调，更未完成 Sim-to-Real 验证。

## 2. 已实施的迭代与结果

| 阶段 | 措施 | 当时的观察或证据 | 结论与边界 |
| --- | --- | --- | --- |
| 工程收敛 | 从多机型、Flat/Rough 收敛到 Lite3 Flat；保留训练、回放、评测链路。 | 当前任务注册只有 Lite3 Flat；历史 Rough 训练不做适配。 | 后续优化集中在同一台有真机可验证的 Lite3。 |
| 可视化与工具 | 调整渲染；修复 PLAY、benchmark 及离线汇总的命令和路径处理。 | `quality` 等尝试曾出现白/灰画面；改为 `balanced + TAA + 4 samples_per_pixel` 后，用户确认可清楚观察，且不需要截图。 | 渲染只用于观察，不作为步态/动力学改进证据。 |
| 初版 Flat（2026-10-05 12:09） | 建立平地训练基线并用 PLAY/benchmark 检查。 | 用户观察到拖地、原地旋转欠佳、对角腿协调不足。日志目录为 `2026-10-05_12-09-18`；旧 benchmark 格式未包含后续全部指标。 | 需要改进摆脚、转向样本和对角步态，不仅看总 reward。 |
| 对角小跑 v2（2026-10-05 14:36） | 加入 0.6 s 对角步态时钟，明确 `FL/HR` 同相、`FR/HL` 同相且两组相差半周期；调整足端离地、滑移、横移及转向相关配置。 | 用户认为步态趋势比上一版好，但 PLAY 时仍见机身偏低、腿部动作偏大。固定评测见下节。 | 步态方向保留，优先解决运动命令触发后的机身下沉。 |
| 站姿 v3（2026-10-05 23:26） | 旧模型默认角改为 `HipY=-0.67`、`Knee=1.34`；Flat 配置将高度目标设为 `0.34 m`，并对低于 `0.32 m` 的机身追加惩罚。 | `model_4999.pt` 已保存。用户后续仍反馈“给运动指令后身子向下低一点”；未找到该 run 的同协议 `benchmark/summary.csv`。 | 不能据此宣称 v3 已解决下蹲。 |
| 官方参数对齐 + 高度优化 v4（2026-10-08 10:30） | 保留工程内 USD 路径，资产默认姿态改为 `HipY=-0.65`、`Knee=1.3`、初始高度 `0.375 m`；`DCMotorCfg` 改为官方使用的 `DelayedPDActuatorCfg`，求解器速度迭代设为 1。Flat 配置提高高度与运动时关节姿态约束，保留 v2 的对角步态设置。 | 新 run `2026-10-08_10-30-48_lite3_flat_new_actuator_v4` 已保存参数快照和 checkpoint；参数快照显示 2048 个环境、目标 5000 次迭代。截至建档，未见该 run 的正式 benchmark 汇总。 | 本次 v4 从头训练；不能把旧 DCMotor/default-pose checkpoint 当作新动力学的效果证明。实际改进待 PLAY 与固定协议评测。 |
| 部署侧模型统一 | 引入云深处 Lite3 部署工程；PyBullet URDF、MuJoCo 楼梯场景及查看器改用共享的 `robot_models`；暂保留 `rl_deploy/Lite3_description/` 场景。 | MuJoCo 楼梯场景可加载且有 12 个执行器，URDF 网格引用齐全；当前 Python 环境缺少 PyBullet，C++ 部分未编译验证。 | 只验证了模型路径和部分加载链路，不能宣称训练策略已完成 Sim-to-Sim 或真机部署。 |

### v2 的量化基线

下表取自 `rl_training/logs/rsl_rl/deeprobotics_lite3_flat/2026-10-05_14-36-54_lite3_flat_diagonal_trot_v2/benchmark/summary.csv`。高度是仿真中机身 `base_height_m_mean`；步态与刮地数据是该版本 benchmark 的统计，**不是 v4 结果**。

| 命令类别 | 平均机身高度 | 步态时钟匹配 | 摆动相触地率 | 支撑脚滑移速度 |
| --- | ---: | ---: | ---: | ---: |
| 站立 | 0.291 m | — | — | 0.052 m/s |
| 前进/后退 | 0.260 m | 0.795 | 20.3% | 0.127 m/s |
| 横移 | 0.261 m | 0.789 | 21.3% | 0.132 m/s |
| 原地转向 | 0.262 m | 0.791 | 25.6% | 0.104 m/s |

运动时比站立低约 0.03 m，和用户的“起步就下蹲”观察一致。表中这些类别的终止率均为 0，但不能据此判定姿态、足端接触或真机安全性合格。v1 与 v2 的 benchmark 字段不同，不把它们的数值差当成严格同协议改进。

## 3. v4 当前生效配置与设计原因

以下数值已与 v4 run 的 `params/env.yaml` 核对：

- **资产**：本工程 `robot_models` 内 Lite3 USD；默认角 `HipX=0`、`HipY=-0.65`、`Knee=1.3`，初始机身高度 `0.375 m`；Hip/Knee 均为 `DelayedPDActuatorCfg`，延迟 0～1 个物理步。它不需要 `DCMotorCfg` 的 `saturation_effort`。不要把官方资产数据目录路径直接替换到本工程。
- **高度**：`base_height_l2=-20`，目标 `0.35 m`；`base_height_below_target=-8`，阈值 `0.34 m`。这是为了增加运动时持续下蹲的代价，而非照搬官方 Rough 配置的 `0.55 m` 高度目标。
- **关节姿态**：`joint_pos_penalty=-0.8`、`stand_still_scale=3.0`。运动时惩罚比 v3 的 `-0.5` 更强；站立时有效系数约为 `-2.4`，接近原先的 `-2.5`。这一改动针对“收到运动命令后约束突然变弱”的问题。
- **步态保留**：`feet_gait=1.2`，周期 `0.6 s`，足端顺序 `FL, FR, HL, HR` 对应偏移 `[0, 0.5, 0.5, 0]`；`feet_height=-5`，世界高度目标 `0.08 m`；`feet_slide=-0.45`、`action_rate_l2=-0.04`。本轮没有通过削弱对角小跑约束来换取机身高度。
- **任务与观测**：前后、横移、原地转向、混合命令各占 25%；actor 不使用仿真才能直接获得的机身线速度或地形高度扫描，critic 可以使用机身线速度。

现有 [训练策略文档](lite3_training_strategy.md) 中的默认角 `-0.67/1.34`、高度奖励 `-12/-4` 和阈值 `0.32 m` 属于 v3 阶段，不是 v4 生效值；`rl_training/README.md` 的训练示例也仍写着 v3。新对话应以代码和 v4 参数快照为准，不要按旧数值继续调参。

## 4. 下一轮实验与记录方法

1. 等 v4 训练完成后，**明确指定 v4 的 checkpoint** 做 PLAY 和 benchmark，避免自动选到别的 run；至少分别检查站立、前后、横移、原地转向和混合命令。
2. 用相同 trials、命令矩阵、warmup 和时长记录 `base_height_m`、三轴速度误差、`gait_clock_match`、对角腿同步/反相、`swing_scuff_rate`、`stance_slip_mps`、倾角与终止率。以 v2 的高度、刮地率作问题参照；由于 v4 同时改变了执行器和默认姿态，不能把两版差异全部归因于某一条 reward。
3. 若 v4 仍下蹲，先判断是整体站高不足、仅速度命令触发下沉，还是某一类命令（尤其原地旋转）引发；再只改一项奖励或一组强耦合参数。不要在没有 benchmark 时继续叠加惩罚。
4. 后续接入 `rl_deploy` 时，逐项核对关节顺序、默认角、动作缩放、50 Hz 控制周期、观测组成与归一化、执行器/延迟假设；Sim-to-Sim 通过前不把策略部署到真机。

后续每版建议追加以下记录，不覆盖历史结论：

```text
日期 / run / checkpoint：
相对上一版的唯一改动及预期机制：
训练参数快照：num_envs、iterations、asset、flat cfg
评测协议与结果：每类命令的均值、方差、失败情况
PLAY 观察：机身高度、四足接触、转向、拖地、动作幅度
结论：已改善 / 退化 / 无法判断；下一步要验证什么
```
