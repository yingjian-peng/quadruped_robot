# Sim-to-Sim

用于验证训练策略在目标仿真器中的行为一致性。后续实现应至少包含模型加载、观测构造、策略推理、动作映射和仿真运行入口。

## 机器人模型可视化

`view_robot.py` 只用于检查 Lite3 的 MuJoCo 模型。它直接加载供应商提供的原生
MJCF：`robot_models/deeprobotics/Lite3/Lite3_mjcf/mjcf/Lite3.xml`；不再扫描、导入或转换
仓库中其它机器人的 URDF。请在 sim-to-sim 环境（`pyj_rl_simtosim`，含 MuJoCo 3.11）下运行，
工作目录为仓库根目录：

```bash
# 无重力检查：鼠标拖动可从各角度查看；右上方 Joint 的 12 个滑动条可直接调整关节角度
conda activate pyj_rl_simtosim
cd /home/robot/pengyingjian_external/quadruped_robot
python rl_deploy/sim_to_sim/view_robot.py

# 重力检查：机器人会在方格地面上落下/接触
python rl_deploy/sim_to_sim/view_robot.py --gravity
```

Lite3 的原生 MJCF 本身已经包含浮动基座、12 个电机、接触几何和视觉网格。方格地面和渐变天空
独立放在 `robot_models/deeprobotics/Lite3/Lite3_mjcf/mjcf/view_environment.xml`。脚本仅在缓存目录
（默认 `~/.cache/quadruped_robot/mujoco/`）生成一个引用原模型和该环境文件的显示入口；原始资产
不会被修改。显示时固定基座高度为 `z=0.50 m`，初始时 12 个关节均为模型默认位置。
脚本保留供应商原始的 12 个 `motor` 力矩控制器，但默认模式不写入 `ctrl`。初始时 12 个关节均为
MJCF 默认值 `0`；右上方 `Joint` 滑动条会直接写入关节 `qpos`，脚本每帧调用 `mj_forward()`，只
重新计算运动学和显示而不改变关节值。下方 `Control` 是电机力矩输入（范围 `-30` 到 `30`），不是
关节角度，在默认模式不会驱动机器人。传入 `--gravity` 后，脚本才调用 `mj_step()` 推进动力学，使
原始电机控制和重力作用于机器人，并让它在地面上落下和接触。
