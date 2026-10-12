# Lite3 平地运动控制交接记录

日期：2026-10-10  
用途：在新对话中继续迭代 Lite3 平地运动策略。本文概括本轮确认的基线、当前源码配置、已完成的修复、评测结论和下一步命令。源码是“现在配置”的依据；每个训练 run 下的 `params/env.yaml` 和 `params/agent.yaml` 才是该 checkpoint 实际训练参数的依据。

2026-10-11 更新：后续优化以 [综合分析与实验计划](lite3_optimization_plan_2026-10-11.md) 为准。下文训练命令为已完成实验的记录；下一轮优先核实评测物理条件及现有低高度惩罚，不直接重复高度权重20实验。

## 1. 当前结论

- 当前运动策略基线为 `model_2998.pt`：`logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual/model_2998.pt`。
- 用户在 PLAY 中确认该策略已能前进、后退、左右平移和原地旋转，整体基本可用。主要可继续优化的是：给运动命令后机身略微下沉、机身水平稳定性、运动时腿向内收。
- 新增配对 GUI 复测显示，`model_3297.pt` 相比 `model_2998.pt` 没有改善：两者各跑了 11 个固定命令、每命令 1 次，在同一套 Play 条件下均无跌倒；但新模型平均根高由约 `0.297 m` 降至 `0.287 m`，roll/pitch 倾角略增，横移误差变大，其他速度与 trot 指标基本持平。结论是不要从 `model_3297.pt` 按原配置继续加训。
- 后续已将高度权重提高到 `-20` 并完成 250 次微调，得到 `model_3247.pt`。11 个 Play 条件命令均无跌倒；运动高度相比 `model_3297` 仅回升约 `1 mm`，相较 `model_2998` 仍低约 `9 mm`，且步态匹配和刮地/滑移指标退化。`model_2998` 目前仍是综合表现最好的父 checkpoint。
- 推荐恢复到 `model_2998.pt` 作为策略初始化，先校准高度目标的奖励强度，再进行有明确单变量假设的小步微调；暂不从随机权重重训。由于新旧模型都能稳定完成默认 Play 条件测试，已有步态和速度跟踪值得保留。
- 用户澄清：此前零动作测试出现的翻倒来自初始落地接触冲击，不能据此断言 PD 无法支撑。评估默认站立时要区分初始冲击、接触分配和策略稳定性；不要把旧零动作试验结论当作已确认根因。

## 2. 当前配置基线

### 机器人资产、初始状态与执行器

配置入口：[deeprobotics.py](../source/robot_lab/quadruped_robot/assets/deeprobotics.py)

- 初始根位置 `(0, 0, 0.375) m`，零初速度，单位四元数。
- 每腿默认关节角 `HipX=0, HipY=-0.65, Knee=1.30 rad`。
- 执行器为 `DelayedPDActuatorCfg`：Hip 力矩/速度上限 `24 N·m / 26.2 rad/s`，Knee `36 N·m / 17.3 rad/s`；训练 `kp=30, kd=1`，延迟 0–1 个物理步。
- `0.375 m` 是每回合出生根高；它不等于运行时高度目标。当前奖励的运行高度目标是 `0.32 m`。不要因为调奖励目标就直接把出生高度也降到 `0.32 m`，否则可能改变初始接触条件。
- 当前 runner 的默认角仍是 `HipY=-0.67, Knee=1.34 rad`，`kp=30, kd=0.5`，与训练资产不完全相同。导出部署前必须明确要对齐哪套执行器/默认角，并用同一模型和 checkpoint 做验证。

### 观测、动作和 PPO

配置入口：[flat_env_cfg.py](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/flat_env_cfg.py)、[rsl_rl_ppo_cfg.py](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/deeprobotics_lite3/agents/rsl_rl_ppo_cfg.py)

- Actor 为 47 维：机身角速度 3、投影重力 3、速度命令 3、关节位置 12、关节速度 12、上一动作 12、步态相位 sin/cos 2。Critic 额外有 base 线速度 3，共 50 维。
- 12 维关节位置增量，关节顺序为 FL、FR、HL、HR，每腿 HipX/HipY/Knee；动作比例 HipX `0.125`，其他 `0.25`。
- 控制频率 50 Hz；步态周期 `0.6 s`。命令包含站立、前后、横移、原地转向和混合模式。
- PPO 每环境采样 24 步；MLP `[512,256,128]`、ELU；5 epochs、4 minibatches、学习率 `5e-4`、adaptive KL `0.01`、`gamma=0.99`、`lambda=0.95`。

### 当前奖励、终止和域随机化

- 当前微调候选：`base_height_l2=-20`，目标 `0.32 m`；低于 `0.32 m` 的保护惩罚权重 `-2`。GUI基线已完成，现从 `model_2998.pt` 开独立验证run。
- 机身姿态：`flat_orientation_l2=-5` 已惩罚 roll/pitch；先观察此项与高度调整的效果，不重复叠加另一条姿态项。垂直速度 `-2`，水平角速度 `-0.05`。
- 步态：对角 trot，FL/HR 同步、FR/HL 同步，相位差半周期；`feet_gait=+1.5`、周期 `0.6 s`、支撑占空比阈值 `0.56`。足端高度项 `-2`（目标 `0.08 m`），腾空时间 `+0.3`（阈值 `0.22 s`）。
- 腿内收：新增 `feet_stance_width=-2`，只惩罚承重足横向中心距小于 `0.13 m` 的情况，不把摆动足轨迹锁到固定位置。
- 小腿接触：只匹配四个 SHANK，`undesired_contacts=-3`；不因小腿接触本身直接终止。
- 跌倒终止：根高低于 `0.18 m` 或倾角超过 `0.8 rad`。`illegal_contact` 关闭。
- 平地基准阶段域随机化收窄：reset x/y `±0.05 m`、z `±0.02 m`、roll/pitch `±0.05 rad`；机身质量 `±0.1 kg`、其他刚体质量倍率 `0.98–1.02`、质心每轴 `±5 mm`、摩擦 `0.95–1.0`、PD 增益倍率 `0.95–1.05`，外力及推扰较小。先得到稳定基线，再按组扩大随机范围。

详见逐项配置说明：[lite3_training_config.md](lite3_training_config.md)。

## 3. 近期配置改动及验证

- 把运行高度目标改为 `0.32 m`，保持出生根高 `0.375 m`。
- 保留原有机身水平惩罚 `-5`；新增支撑脚横向宽度惩罚，缓解运动中腿向内收。
- 保持 trot 时钟约束，目标是在保留已形成的运动能力时改善姿态和腿部支撑宽度。
- 新增宽度奖励时遇到张量形状问题：Isaac Lab 的 `quat_apply_inverse` 不会自动广播 `[N,1,4]` 与 `[N,4,3]`。代码现将根四元数显式扩展到每个足端。使用 Isaac Lab 数学函数检查了 256 环境 × 4 足输入形状。
- 随后用 4 环境完成从 `model_2998.pt` 恢复、一次环境 step 和一次 PPO 更新的 smoke run，确认奖励可以执行。它只证明配置和训练流程可运行，不代表训练效果改善。
- 续训启动诊断已加入 `train.py`：会打印环境创建、runner 构建和首个向量化 `env.step()` 的进度；`--debug_startup` 在首个 step 卡超过 30 秒时转储 Python 线程栈。此前用户看到 `[INFO] Policy weights restored...` 后无输出时，程序尚未证明已经进入首个有效训练更新。
- GUI benchmark 已修正终止环境自动 reset 后造成的假阴性跌倒统计；移除运动命令后会继续用零速度命令运行策略维持站姿，默认保持 5 秒。默认保留 Isaac GUI；测试中途结束命令后不会直接放开关节让机器人塌下去。

## 4. 续训命令

从 `model_2998.pt` 再训练 250 次，保存为独立的新 run。当前源码已将 `base_height_l2.weight` 设为 `-20`：

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --headless --num_envs 2048 --max_iterations 250 \
  --resume \
  --load_run 2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual \
  --checkpoint model_2998.pt --reset_optimizer --resume_full_command_range \
  --run_name lite3_flat_h032_heightw20_ft250 \
  --debug_startup
```

`--max_iterations 250` 表示在 checkpoint 上追加 250 次更新。`--reset_optimizer` 保留策略权重并重置 PPO 优化器；`--resume_full_command_range` 从完整命令范围继续；新 run 独立保存。首轮若正常开始更新，后续可去掉 `--debug_startup`。

## 5. GUI 评测建议

评测脚本：[benchmark.py](../scripts/reinforcement_learning/rsl_rl/benchmark.py)。不传 `--headless` 会显示 Isaac GUI；不指定 `--case` 会评固定命令矩阵。要单独肉眼观察一个命令，可指定 case 并只跑 1 次：

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/benchmark.py \
  --task QuadrupedRobot-Velocity-Flat-Deeprobotics-Lite3-v0 \
  --checkpoint_path logs/rsl_rl/deeprobotics_lite3_flat/<新run目录> \
  --case forward --trials 1 --duration_s 8 --warmup_s 2 \
  --nominal_conditions
```

将 `forward` 依次换成 `stand`、`backward`、`left`、`right`、`ccw`、`cw`、`left_ccw`、`right_cw`、`forward_ccw`、`forward_cw`。GUI 评测结束后脚本默认继续用策略维持零命令站姿 5 秒；可用 `--hold_window_s` 调整。

续训前后使用同样的命令、随机条件和时长比较：

- 跌倒次数、根高均值、绝对 roll/pitch、速度 MAE。
- 对角 trot 相位匹配、足端接触占空比和摆动刮地率。
- 支撑阶段左右足距及四足、小腿接触占空比/接触力。
- 肉眼确认命令切换时高度是否下沉、身体是否保持水平、腿是否仍向中线收。

不要只依据总 reward 或单张画面判断。固定命令矩阵的汇总日志可能混有不同日期、配置和 benchmark 版本；对比时优先使用本次新 run 输出的 CSV，并核查对应 `params/env.yaml`。

## 6. 资产、官网包和部署接口记录

- 用户提供的官网训练包位于 `/home/robot/pengyingjian_external/rl_training`。该包使用 Isaac Sim 5.1 / Isaac Lab 2.3.2；本工程环境是 Isaac Sim 4.5 / Isaac Lab 2.3.0。决定是沿用本工程框架和奖励，参考/采用 Lite3 官方资产与执行器基线，不整体替换训练框架。
- USD、URDF、MJCF 的碰撞表达曾被反复检查；三种资产的网格/碰撞表示并非完全一致，足端视觉 prim 还出现过指向 `Lite3_physics.usd` 的 unresolved reference 警告。环境和 PPO smoke run 通过，警告目前应作为资产清理事项单独核实，不要误当成训练卡住的充分证据。
- 过去的零动作和 MuJoCo 静态测试提示小腿碰撞可能接近/低于足底支撑平面，但这些测试的初始接触状态对结果敏感。后续若重新调查支撑几何，应记录初始化冲击、每条足端/小腿法向力和根姿态，并做单变量对照；不要与本轮奖励微调混在一起。
- 部署 runner 已支持读取 ONNX 输入大小 45（旧策略）或 47（包含 gait phase），47 维时提供相位输入；`last_action` 在首次推理前置零；支持指定模型路径并检查输入维度。测试基于旧/新 ONNX 的 smoke test 曾通过。部署默认角和 `kd` 仍与训练资产不同，任何新策略部署前都应重新核对观测顺序、命令缩放、频率、默认角、PD 增益和 ONNX 模型版本。

## 7. model_3297 配对评测与决策

评测脚本默认 GUI、Play 随机化条件；11 个固定命令并行，每个命令 1 个 rollout，时长 8 s、前 2 s 为预热。测试结果保存在临时目录，`/tmp` 可能被系统清理：

- 新模型：[rollouts.csv](../logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_16-12-49_lite3_flat_h032_stancewidth_ft300/benchmark_play_gui/rollouts.csv)、[summary.csv](../logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_16-12-49_lite3_flat_h032_stancewidth_ft300/benchmark_play_gui/summary.csv)。
- 旧基线：[rollouts.csv](../logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual/benchmark_play_gui/rollouts.csv)、[summary.csv](../logs/rsl_rl/deeprobotics_lite3_flat/2026-10-10_14-18-29_lite3_flat_trot_h0375_to3000_manual/benchmark_play_gui/summary.csv)。
- `model_2998` → `model_3297`：11/11 → 11/11 无跌倒；高度均值 `0.2973` → `0.2867 m`；倾角均值 `0.0289` → `0.0348 rad`；MAE `(vx,vy,wz)` 为 `(0.0409,0.0235,0.0336)` → `(0.0422,0.0309,0.0338)`；gait clock match `0.9733` → `0.9676`；支撑滑移约 `0.0320` → `0.0323 m/s`；关节功率约 `27.94` → `29.25 W`。新模型没有收益，主要退化是高度和横移跟踪。
- TensorBoard 新续训 run 最后几十步的平均回报约 `91.9`、平均 episode length 约 `995/1000`，策略仍稳定；但回报不能和父 run 直接比较，因为训练目标改过。`Episode_Reward/feet_stance_width` 约 `-0.05`，说明新项有被执行；benchmark 没记录支撑足横向距离，因此不能判断腿内收是否改善。
- 相同 checkpoint 在 `--nominal_conditions` 的 11 环境并行 GUI 测试中都很早触发倾角终止；同日标准 Play 条件测试则都无跌倒。这个差异提示初始化接触条件对结果敏感，名义条件结果不能单独作为策略退化证据。以后要比较时先固定并报告评测 preset，再决定是否做无随机化的接触诊断。

奖励解释：本轮把 `base_height_l2` 的目标从 `0.375` 下调到 `0.32 m`，同时把 `base_height_below_target` 从 `(target=0.32, weight=-4)` 改成 `(target=0.29, weight=-2)`。前者是对目标高度的对称平方误差，后者在阈值以下才生效；这组变化整体减弱了策略保持较高机身的压力，和“给运动命令时减少下沉”的目标不够一致。仅把 target 写成 `0.32 m` 并不保证策略会达到 `0.32 m`。

决策：

1. 不从头重训，也不以 `model_3297.pt` 作为下一轮父 checkpoint；回到 `model_2998.pt`。
2. GUI基线已补测。当前从 `model_2998.pt` 开独立微调 run，仅将低高度保护阈值从 `0.29 m` 调到 `0.32 m`，其他奖励/PPO参数不变，追加250次更新。该候选尚未训练验证；不要同一轮再加姿态项或大改步态奖励。
3. 支撑脚宽度项先维持现值。评测增加每个时刻的支撑左右足距、最小足距和均值后，再用视频与数据判断其作用；若决定增强宽度惩罚，另开一个从同一 `model_2998.pt` 出发的单变量对照。
4. 每轮沿用标准 Play 条件 GUI 固定命令矩阵，最好增加到每命令 3 次试验；比较根高、roll/pitch、跌倒、速度 MAE、gait match、关节功率及支撑脚宽度。确认指标没有提升时回到父 checkpoint，不要为训练迭代数继续加训。

## 8. 旧实验如何使用

本轮目标是从用户确认可用的 `model_2998.pt` 继续优化。v4–v7、旧 v3.1、不同资产/执行器组合下的旧 benchmark 只用于理解历史，不作为当前策略优劣的直接对照。尤其是旧 benchmark CSV 曾存在自动 reset 后跌倒标记错误，且不同版本字段/条件不一致；横向比较前先确认评测代码版本和 `params/env.yaml`。

新对话建议保留 `model_2998.pt` 为当前最佳父 checkpoint。不要盲目继续增大全局高度权重；补充 benchmark 的支撑足宽度指标。此前建议新增运动高度项，现修正为先检验现有单侧低高度保护；仅增加运动门控不能自动解决奖励间的竞争。`model_3297.pt`、`model_3247.pt` 和对应 CSV 均作为对照保留，不覆盖旧 run。

## 9. model_3247 复测结论

标准 Play 条件 GUI 测试的 11 个命令每项 1 次、8 s（前 2 s warmup），结果为 0/11 跌倒。运动命令下 `model_2998 → model_3297 → model_3247`：根高 `0.2964 → 0.2860 → 0.2870 m`，倾角 `0.0295 → 0.0328 → 0.0326 rad`；线速度 MAE `0.0450 → 0.0464 → 0.0345`，横向 MAE `0.0259 → 0.0340 → 0.0359`，偏航 MAE `0.0366 → 0.0370 → 0.0376`；时钟匹配 `0.9733 → 0.9676 → 0.9581`；足端刮地率 `0.0292 → 0.0303 → 0.0509`。站立根高约 `0.307 m`，收到命令后仍降至约 `0.287 m`。

两次微调的训练时长和轨迹不同，每命令也只测了一次，约1 mm的差异不足以证明提高高度权重产生了抬高作用。`model_3247` 同时伴随步态/接触质量回退，不应当作整体更优 checkpoint；确认候选前需重复评测。结果已保存到 Lite3 各自训练 run 下的 `benchmark_play_gui/`。
