# rl_deploy

DeepRobotics LITE3 强化学习策略部署子工程。本目录用于逐步实现训练策略从 Isaac Lab 到其他
仿真器和 LITE3 真机的部署链路。

## Planned scope

- `sim_to_sim/`：策略导出后的跨仿真器验证，例如 Isaac Sim 到 MuJoCo。
- `sim_to_real/`：LITE3 实机通信、状态预处理、策略推理、动作后处理和安全保护。
- 训练与部署共同使用仓库根目录 `robot_models/` 中的机器人模型。

当前目录仅提供工程骨架；尚未包含可运行的实机控制程序。部署实现应明确记录策略观测顺序、动作
顺序、归一化参数、50 Hz 控制频率和 LITE3 关节映射，避免与训练配置不一致。
