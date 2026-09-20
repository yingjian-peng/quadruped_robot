# Unitree Go2 强化学习框架分析

本文只分析本工程中的 **Unitree Go2 腿式四足机器人**，不覆盖 Go2W、B2、A1 或 DeepRobotics 系列。

## 先读结论

这是一个基于 **Isaac Lab Manager-Based 环境** 和 **RSL-RL PPO** 的四足速度跟踪训练框架。

- 任务目标：给定机体坐标系下的速度/朝向指令，学习稳定行走与速度跟踪，并偏向对角小跑（trot）。
- 策略输出：12 维关节位置目标增量，不直接输出力矩或足端位置。
- rough 环境：训练机器人适应楼梯、障碍、粗糙地、坡面，并启用地形难度课程。
- flat 环境：只在无限平地训练，不启用课程学习。
- 策略（actor）只用本体感觉、命令和上一帧动作；rough 环境的高度扫描只给 critic，因此这是非对称 actor-critic。
- 它不是导航任务：没有目标点、路径规划或 actor 的前视地形感知。

## 一、框架总览

```text
Gym task ID
  → Go2 Flat / Rough 配置
  → 通用速度环境（Scene + MDP）
  → Isaac Lab ManagerBasedRLEnv
  → RslRlVecEnvWrapper
  → RSL-RL OnPolicyRunner / PPO
```

主要代码位置：

- [Go2 任务注册](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/__init__.py)
- [Go2 rough 配置](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/rough_env_cfg.py)
- [Go2 flat 配置](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/flat_env_cfg.py)
- [通用环境与 MDP 配置](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- [训练入口](../scripts/reinforcement_learning/rsl_rl/train.py)
- [PPO 配置](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/agents/rsl_rl_ppo_cfg.py)

工程为 Go2 注册了两个 Gym 环境：

```text
RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0
RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0
```

## 二、训练过程（train）

以 rough Go2 为例：

```bash
cd /home/robot/pengyingjian_external/quadruped_robot/rl_training

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless --num_envs 4096
```

`train.py` 的执行过程如下：

1. 根据 task ID 从 Gym registry 取得 Go2 的环境配置和 RSL-RL PPO 配置。
2. 创建 Isaac Lab `ManagerBasedRLEnv`，默认创建 4096 个并行环境；`--num_envs` 可覆写数量。
3. 用 `RslRlVecEnvWrapper` 将 Isaac Lab 环境转换为 RSL-RL 可用的向量环境。
4. 创建 RSL-RL 的 `OnPolicyRunner`；如果传入 `--resume`，则加载已有 checkpoint。
5. 每个环境并行采样 24 个控制步，形成一个 PPO rollout。
6. 对 rollout 计算 value、GAE advantage 与 return，再执行 PPO 多轮更新。
7. 按固定间隔保存模型、TensorBoard event、实际 `env.yaml` 与 `agent.yaml`。

时间尺度：

| 项目 | 设置 | 含义 |
|---|---:|---|
| 物理步长 | 0.005 s | 200 Hz PhysX 仿真 |
| `decimation` | 4 | 每个策略动作执行 4 个物理步 |
| 控制步长 | 0.02 s | 策略控制频率 50 Hz |
| episode 长度 | 20 s | 最多 1000 个控制步 |
| rollout | 24 step/env | 每个环境每轮采样 0.48 s |
| 默认并行环境数 | 4096 | 每轮共 98,304 条 transition |

训练迭代数：rough 默认 20,000，flat 默认 5,000；每 100 iteration 保存一次。

> `agents/cusrl_ppo_cfg.py` 虽然保留在工程中，但当前 `scripts/reinforcement_learning/rsl_rl/train.py` 不读取它；本训练入口实际使用的是 RSL-RL。

训练产物位于：

```text
logs/rsl_rl/unitree_go2_{rough|flat}/<timestamp>/
├── model_*.pt
├── events.out.tfevents.*
└── params/
    ├── env.yaml
    └── agent.yaml
```

## 三、Go2 机器狗设置

机器人资产定义在 [assets/unitree.py](../source/robot_lab/quadruped_robot/assets/unitree.py)。

| 项目 | Go2 设置 |
|---|---|
| URDF | `robot_models/unitree/go2_description/urdf/go2_description.urdf` |
| 自由度 | 12 个腿部关节 |
| 关节顺序 | FR、FL、RR、RL，每条腿 hip/thigh/calf |
| 初始 base 高度 | 0.38 m |
| 默认站姿 | hip=0，thigh=0.8 rad，calf=-1.5 rad |
| PD stiffness | 25 |
| PD damping | 0.5 |
| 力矩上限 | 23.5 |
| 关节速度上限 | 30 rad/s |
| 软关节限位系数 | 0.9 |
| 自碰撞 | 关闭 |
| 接触传感器 | 开启 |

Go2 的 12 个被控制关节是：

```text
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint
```

## 四、Isaac Lab Scene 设置

### 4.1 rough 场景

rough 场景默认有 4096 个并行环境，环境间距为 2.5 m。地形由 Isaac Lab 的 `ROUGH_TERRAINS_CFG` 生成：每个 tile 为 8 m × 8 m，包含：

- 金字塔上楼梯；
- 金字塔下楼梯；
- 随机箱体障碍；
- 随机高度场粗糙地；
- 金字塔上坡；
- 金字塔下坡。

默认地面物理材质：static friction=1.0、dynamic friction=1.0、restitution=1.0。机器人自身材质会在启动时进行域随机化。

传感器：

| 传感器 | 配置 | 用途 |
|---|---|---|
| `contact_forces` | 覆盖 robot 的所有 link；3 帧历史；记录离地时间 | 接触奖励、滑动惩罚、步态奖励 |
| `height_scanner` | base 上方 20 m 向下；范围 1.6 × 1.0 m；0.1 m 分辨率 | rough 时只给 critic 地形高度图 |
| `height_scanner_base` | 范围 0.1 × 0.1 m；0.05 m 分辨率 | 通用 base 高度奖励的支持项；Go2 最终未启用此奖励 |

高度扫描主网格是 17 × 11，即 187 个射线点。

### 4.2 flat 与 rough 的区别

flat 继承 rough，再进行以下修改：

- 地形从 procedural generator 改为无限 plane；
- 删除两个 height scanner；
- policy 与 critic 都删除 `height_scan`；
- 删除 `terrain_levels` 地形课程。

## 五、MDP：Command、Action、Observation

### 5.1 Command 设置

命令名为 `base_velocity`，本质为机体坐标系中的 `[v_x, v_y, \omega_z]`：

| 项目 | 范围/设置 |
|---|---|
| 前向速度 `v_x` | [-1.0, 1.0] m/s |
| 横向速度 `v_y` | [-1.0, 1.0] m/s |
| yaw 角速度 `ω_z` | [-1.0, 1.0] rad/s |
| heading 目标 | [-π, π] |
| 命令重采样间隔 | 10 s |
| 站立环境比例 | 2% |
| heading 环境比例 | 100% |
| heading 控制刚度 | 0.5 |

命令生成器还会将平面线速度模长小于 0.2 的样本置为零。因此零命令不是一个偶然的小速度样本，而是明确的“站立”训练情形。

`rel_heading_envs=1.0` 表示所有环境都启用 heading command：yaw 命令会由当前朝向与 heading target 的误差通过比例控制产生，并受 `[-1, 1] rad/s` 限制。

> 自定义命令生成器中还有针对 `pits` 地形的“仅允许前进”逻辑，但当前 `ROUGH_TERRAINS_CFG` 没有 `pits` 子地形，因此该分支目前不会触发。

### 5.2 Action 设置

策略动作是 12 维绝对关节位置目标的增量：

```text
q_target = q_default + scale × policy_action
```

| 关节 | action scale |
|---|---:|
| hip | 0.125 rad |
| thigh、calf | 0.25 rad |

动作随后由 Go2 的 PD actuator 转为关节力矩并受电机力矩上限限制。

当前 RSL-RL wrapper 的 `clip_actions=None`；动作 term 内部的裁剪范围为 `[-100, 100]`，实际非常宽。因此策略输出并没有通常意义上的 `[-1, 1]` 强制截断。若训练出现动作爆发或不稳定，这是值得优先检查的配置点。

### 5.3 Observation 设置

Go2 rough 使用非对称 actor-critic：actor 使用可部署的本体感觉，critic 在训练时拥有更多地形信息。

**policy / actor：45 维**

| 观测项 | 维度 | 噪声/缩放 |
|---|---:|---|
| base 角速度 | 3 | 噪声 ±0.2，scale=0.25 |
| projected gravity | 3 | 噪声 ±0.05 |
| 当前速度命令 | 3 | 无噪声 |
| 相对默认关节位置 | 12 | 噪声 ±0.01 |
| 相对关节速度 | 12 | 噪声 ±1.5，scale=0.05 |
| 上一帧 action | 12 | 无噪声 |

**critic：rough 为 235 维，flat 为 48 维**

critic 包含 actor 对应的状态，并额外拥有：

- base 线速度：3 维；
- rough 的高度扫描：187 维。

critic 无观测噪声。Go2 rough 的 policy 显式删除了 `base_lin_vel` 和 `height_scan`，因此策略不能在动作前直接“看到”前方地形，只能从姿态、角速度、关节状态等反馈中适应地形。

## 六、MDP：Events 与域随机化

### 启动时（startup）

| 随机化项 | 范围 |
|---|---|
| 所有刚体静摩擦 | [0.3, 1.0] |
| 所有刚体动摩擦 | [0.3, 0.8] |
| restitution | [0.0, 0.5] |
| base 质量 | 加上 [-1.0, 3.0] |
| 非 base link 质量 | 乘以 [0.7, 1.3] |
| base 质心偏移 x/y/z | 各 ±0.05 m |

### 每次 reset

| 随机化项 | Go2 设置 |
|---|---|
| base x/y 偏移 | ±0.5 m |
| base z 偏移 | [0.0, 0.2] m |
| base roll/pitch/yaw | 各 ±π |
| 初始线/角速度 | 各轴 ±0.5 |
| 关节位置 | 默认位置的 1.0 倍，即无位置扰动 |
| 关节速度 | 0 |
| PD stiffness、damping | 各自随机缩放 [0.5, 2.0] |
| base 随机外力 | 各方向 [-10, 10] |
| base 随机力矩 | 各方向 [-10, 10] |

### 训练中（interval）

每隔 10–15 秒，使用速度设定方式给 robot 一个扰动：x/y 方向均在 `[-0.5, 0.5] m/s`。

这些随机化的目的，是降低策略对单一仿真模型、单一初始姿态和固定 PD 参数的依赖，提高 sim-to-real 鲁棒性。

## 七、MDP：Reward 设置

Go2 rough 在配置末尾会删除所有权重为 0 的 reward term。下表是实际启用的奖励；Isaac Lab 计算时会再乘控制步长 `dt=0.02`，因此表中权重可理解为每秒尺度。

| 类别 | Reward term | 权重 | 作用 |
|---|---|---:|---|
| 主任务 | `track_lin_vel_xy_exp` | +3.0 | 指令与实际 xy 线速度的指数跟踪奖励 |
| 主任务 | `track_ang_vel_z_exp` | +1.5 | 指令与实际 yaw 角速度的指数跟踪奖励 |
| 稳定 | `upward` | +1.0 | 鼓励 base 保持正立 |
| 步态 | `feet_gait` | +0.5 | FL-RR、FR-RL 对角同步，形成 trot 倾向 |
| 步态 | `feet_air_time` | +0.1 | 移动时鼓励有效腾空与跨步 |
| 静止 | `feet_contact_without_cmd` | +0.1 | 零命令时鼓励脚接触地面 |
| 稳定 | `lin_vel_z_l2` | -2.0 | 惩罚竖直速度 |
| 稳定 | `ang_vel_xy_l2` | -0.05 | 惩罚 roll/pitch 角速度 |
| 能耗 | `joint_torques_l2` | -2.5e-5 | 惩罚关节力矩平方 |
| 能耗 | `joint_power` | -2e-5 | 惩罚绝对机械功率 |
| 平滑 | `joint_acc_l2` | -2.5e-7 | 惩罚关节加速度 |
| 平滑 | `action_rate_l2` | -0.01 | 惩罚相邻动作变化 |
| 姿态 | `joint_pos_penalty` | -1.0 | 惩罚偏离默认站姿；静止时放大 5 倍 |
| 姿态 | `stand_still` | -2.0 | 零命令时惩罚关节偏离默认位置 |
| 对称 | `joint_mirror` | -0.05 | 惩罚 FR-RL、FL-RR 的关节不对称 |
| 安全 | `joint_pos_limits` | -5.0 | 惩罚越过软关节位置限位 |
| 接触 | `undesired_contacts` | -1.0 | 惩罚 calf 以外的 link 接触 |
| 接触 | `contact_forces` | -1.5e-4 | 惩罚 calf 接触力过大 |
| 接触 | `feet_slide` | -0.1 | 惩罚接触脚横向滑动 |
| 步态 | `feet_air_time_variance` | -1.0 | 惩罚四足腾空/接触时长不均衡 |
| 摆腿 | `feet_height_body` | -5.0 | 移动时惩罚摆脚相对机身高度偏离 -0.2 m |

主要 reward 的含义：

- 速度跟踪项采用指数核：误差越小，奖励越接近 1；同时会根据机身是否接近正立进行门控。
- `feet_gait` 让 FL-RR 与 FR-RL 分别同步，而两组之间反相，从而偏向对角小跑。
- `joint_mirror`、`feet_air_time_variance` 与 gait reward 一起约束出更规整、左右对称的步态。
- 能耗、动作变化、关节加速度、足端滑动共同抑制不平滑和不经济的动作。

奖励具体实现见 [mdp/rewards.py](../source/robot_lab/quadruped_robot/tasks/manager_based/locomotion/velocity/mdp/rewards.py)。

## 八、MDP：终止条件

Go2 实际启用的终止条件较少：

| 条件 | 是否为 time-out | 说明 |
|---|---|---|
| `time_out` | 是 | episode 达到 20 s |
| `terrain_out_of_bounds` | 是 | rough 时接近整张 terrain map 边界 3 m；plane 上始终为 false |
| `illegal_contact` | 否 | **Go2 中显式设置为 `None`，未启用** |

因此，base 落地、髋部接触地面、翻倒等不会导致立即 reset；它们主要通过姿态、速度跟踪和接触类奖励变差受到惩罚。这个设计不同于许多“跌倒即终止”的腿式 locomotion 环境。

## 九、课程学习：是否启用、学习什么

### rough：启用地形课程

Go2 rough 保留 `terrain_levels`：

- 初始最大地形等级为 5；
- 如果机器人在一个 episode 内走出 tile 一半以上距离（大于 4 m），则下次被分配到更难 terrain level；
- 如果实际走的距离小于命令速度所要求距离的一半，则降低地形等级；
- 难度提升体现在楼梯高度、障碍、粗糙度和坡面等地形参数上。

这不是额外学习一个“课程目标”，而是让已经能完成当前地形速度跟踪的环境逐渐面对更困难的地面条件。

### 命令课程：实现存在，但 Go2 未启用

通用环境中实现了线速度和 yaw 命令范围课程：当跟踪奖励足够高时，命令范围会从原范围的 10% 扩展至 100%。但是 Go2 rough 明确将：

```python
self.curriculum.command_levels_lin_vel = None
self.curriculum.command_levels_ang_vel = None
```

因此 Go2 从训练开始就面对完整的 `[-1, 1]` 指令范围。

### flat：无课程学习

Go2 flat 删除了 `terrain_levels`，而速度命令课程本来就被 Go2 禁用，所以 flat 没有启用课程学习。

## 十、最终理解：Go2 到底学什么

rough Go2 所学的是：

> 在全速度指令范围内，通过 12 维关节位置 PD 控制，让 Go2 保持正立、少滑动、少耗能、关节动作平滑，并稳定地完成前进、横移、转向和站立；在 rough 任务中，还要逐渐适应更难的程序化地形。

它没有学习：

- 导航到指定空间目标；
- 路径规划；
- 直接力矩控制；
- actor 侧的前视地形图感知；
- Go2W 等轮足机器人的轮式控制逻辑。

