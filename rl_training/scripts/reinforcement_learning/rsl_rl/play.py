"""播放训练好的策略，或将策略导出为 ONNX 格式。"""


import argparse                                         # 解析播放和导出所需的命令行参数
import os                                               # 处理路径拼接、目录创建等操作
import re                                               # 使用正则表达式解析 checkpoint 的迭代编号
import sys                                              # 将当前脚本目录加入模块搜索路径
from pathlib import Path                                # 使用 Path 处理文件系统路径
from isaaclab.app import AppLauncher                    # 启动 Isaac Sim 应用

sys.path.append(os.path.dirname(__file__))              # 让 Python 能导入同目录下的命令行辅助模块
import cli_args                                         # 复用训练脚本使用的通用命令行参数和本地默认值
PROJECT_ROOT = (Path(__file__).resolve().parents[3])    # 根据本文件位置计算 rl_training 项目根目录

# 创建播放与模型导出的命令行解析器
parser = argparse.ArgumentParser(description="Play or export an Isaac Lab RSL-RL checkpoint.")
parser.add_argument("--task", required=True, help="Registered Gymnasium task ID.")
parser.add_argument("--checkpoint_path", type=str, default=None,
    help="Checkpoint file or run directory; defaults to the newest checkpoint under logs/rsl_rl.")  # 说明可传模型文件或运行目录，以及默认查找规则
parser.add_argument("--export_onnx", action="store_true", help="Export ONNX policy files from checkpoint(s) and exit without opening the UI.")

AppLauncher.add_app_launcher_args(parser)               # 加入 Isaac Sim 的 headless、设备等启动参数
args_cli = parser.parse_args()                          # 解析命令行参数
if args_cli.export_onnx:                                # 导出模型不需要图形界面，因此强制启用无头模式
    args_cli.headless = True                            # 设置 Isaac Sim 的 headless 启动标记
cli_args.apply_local_app_defaults(args_cli)             # 补齐设备、渲染和 Kit GPU 参数等本地默认设置
app_launcher = AppLauncher(args_cli)                    # 根据解析完成的参数启动 Isaac Sim

simulation_app = app_launcher.app                       # 保存仿真应用实例，供程序结束时关闭



# ==================================================================================
# 固定初始状态并关闭播放阶段不需要的外力随机化
# ==================================================================================
def stabilize_play_reset(env_cfg) -> None:
    if hasattr(env_cfg.events, "randomize_reset_base"):     # 仅当环境包含基础状态随机化项时才修改它
        env_cfg.events.randomize_reset_base.params = {      # 设置基础位置和姿态的随机偏移范围
            "pose_range": {                                 # 固定初始位置偏移，不随机平移或旋转机器人
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):    # 检查环境是否配置了随机外力事件
        env_cfg.events.randomize_apply_external_force_torque = (None)       # 禁用随机外力和外力矩，避免播放时受到训练扰动
    if hasattr(env_cfg.events, "randomize_push_robot"):                     # 检查环境是否配置了随机推动事件
        env_cfg.events.randomize_push_robot = (None)                        # 禁用随机推动，保证回放更稳定且易于复现



# ==================================================================================
# 设置键盘遥控使用的命令生成方式
# ==================================================================================
def configure_keyboard_command(env_cfg) -> None:
    env_cfg.commands.base_velocity.debug_vis = True             # 开启速度命令的可视化标记
    env_cfg.commands.base_velocity.heading_command = (False)    # 使用直接角速度命令，不采用目标朝向控制模式
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0       # 禁止环境自动生成朝向命令
    env_cfg.commands.base_velocity.rel_standing_envs = (0.0)    # 禁止自动分配站立命令环境，以便键盘接管速度命令
    env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)       # 将自动重采样时间设得很长，避免覆盖键盘输入的命令
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):  # 仅当配置中有超时终止项时才修改它
        env_cfg.terminations.time_out = (None)                  # 播放时不因训练回合时长限制而自动重置机器人



# ==================================================================================
# 设置播放视口中机器人的观察位置和视角
# ==================================================================================
def configure_viewer(env_cfg) -> None:
    env_cfg.viewer.env_index = 0                    # 使用第 0 个环境作为可视化目标
    env_cfg.viewer.eye = (1.6, -1.6, 0.95)          # 设置相机位置，使 Lite3 在视口中足够大，便于检查步态
    env_cfg.viewer.lookat = (0.0, 0.0, 0.28)        # 设置相机注视点，聚焦机器人机身附近
    env_cfg.viewer.origin_type = "world"            # 使用世界坐标系解释相机位置



# ==================================================================================
# 按当前工作目录或 rl_training 根目录解析输入路径
# ==================================================================================
def resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()                 # 展开 ~ 等用户目录简写，并转换为 Path 对象
    if path.is_absolute():                          # 绝对路径不依赖调用目录，直接规范化后返回
        return path.resolve()                       # resolve() 规范化路径中的 .. 等片段
    cwd_path = (Path.cwd() / path).resolve()        # 先尝试将相对路径解释为相对于当前工作目录
    if cwd_path.exists():                           # 如果该路径存在，就优先采用调用者指定的工作目录路径
        return cwd_path                             # 返回当前工作目录下找到的路径
    return (
        PROJECT_ROOT / path
    ).resolve()                                     # 当前目录下不存在时，将相对路径解释为相对于 rl_training 根目录



# ==================================================================================
# 从 checkpoint 文件名中提取训练迭代编号
# ==================================================================================
def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)     # 只接受完整匹配 model_<数字>.pt 格式的文件名
    return (int(match.group(1)) if match else -1)           # 返回文件名中的迭代数；格式不匹配时返回 -1



# ==================================================================================
# 枚举目录中的模型文件并按迭代号排序
# ==================================================================================
def checkpoints_in(directory: Path) -> list[Path]:
    checkpoints = list(directory.glob("model_*.pt"))        # 查找目录顶层符合 model_*.pt 的 checkpoint 文件
    return sorted(                                          # 先按迭代数排序，再用修改时间解决同迭代号的排序关系
        checkpoints, key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime)
    )



# ==================================================================================
# 解析需要播放或导出的 checkpoint 列表
# ==================================================================================
def resolve_checkpoints(agent_cfg, num_checkpoints: int = 1) -> list[str]:
    if args_cli.checkpoint_path is not None:                # 用户明确指定路径时，按该路径查找模型
        checkpoint_path = resolve_input_path(args_cli.checkpoint_path)      # 解析绝对路径或相对路径
        if checkpoint_path.is_dir():                        # 如果输入是运行目录，就枚举其中的模型文件
            checkpoints = checkpoints_in(checkpoint_path)   # 获取该运行目录中按迭代号排序的 checkpoint
            if not checkpoints:                             # 运行目录没有模型文件时，报告清楚的错误路径
                raise FileNotFoundError(                    # 抛出文件不存在错误，终止后续加载流程
                    f"No model_*.pt checkpoint found in: {checkpoint_path}")
            return [str(path) for path in checkpoints[-num_checkpoints:]]  # 返回排序后最新的 num_checkpoints 个模型路径
        if (not checkpoint_path.is_file()):                 # 输入既不是目录也不是已有普通文件时，报告错误
            raise FileNotFoundError(                        # 抛出文件不存在错误，并显示解析后的路径
                f"Checkpoint file does not exist: {checkpoint_path}")
        return [str(checkpoint_path)]                       # 输入是具体文件时，直接返回该模型路径
    log_root = (PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name)  # 未指定路径时，在本任务的日志根目录下自动查找模型
    checkpoints = sorted(                                   # 递归查找所有 checkpoint，按修改时间和迭代号排序
        log_root.rglob("model_*.pt"),
        key=lambda path: (path.stat().st_mtime, checkpoint_iteration(path)),
    )
    if not checkpoints:                                     # 没有找到任何模型时，报告搜索目录
        raise FileNotFoundError(f"No model_*.pt checkpoint found under: {log_root}")  # 抛出文件不存在错误，避免后续对空列表取值
    return [str(path.resolve()) for path in checkpoints[-num_checkpoints:]]  # 返回最近 num_checkpoints 个模型，并转成绝对路径字符串



# ==================================================================================
# 解析单个 checkpoint，并返回排序后最新的一个
# ==================================================================================
def resolve_checkpoint(agent_cfg) -> str:
    return resolve_checkpoints(agent_cfg)[-1]               # 复用多模型查找逻辑，取结果列表的最后一项



# ==================================================================================
# 将当前策略导出为 ONNX 文件
# ==================================================================================
def export_policy_to_onnx(runner, checkpoint: str) -> str:
    export_model_dir = os.path.join(os.path.dirname(checkpoint), "exported")  # 将导出文件放在 checkpoint 同级的 exported 子目录
    filename = (os.path.splitext(os.path.basename(checkpoint))[0] + ".onnx")  # 将模型文件名后缀从 .pt 改成 .onnx
    if hasattr(runner, "export_policy_to_onnx"):            # 优先使用当前 RSL-RL runner 自带的 ONNX 导出接口
        runner.export_policy_to_onnx(path=export_model_dir, filename=filename)  # 传入输出目录和文件名执行导出
    else:                                                   # 兼容没有 runner 导出方法的 RSL-RL 版本
        from isaaclab_rl.rsl_rl import (export_policy_as_onnx)  # 导入 Isaac Lab 提供的通用 ONNX 策略导出函数
        policy_nn = (                                       # 新版本可能把策略保存在 policy，旧版本则可能叫 actor_critic
            runner.alg.policy                               # 优先读取 policy 属性
            if hasattr(runner.alg, "policy")                # 若无 policy 属性，则从 actor_critic 读取策略网络
            else runner.alg.actor_critic
        )
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)  # 读取 actor 的观测归一化器；策略没有归一化器时返回 None
        export_policy_as_onnx(                              # 导出策略网络及其观测归一化器
            policy=policy_nn,                               # 传入需要导出的策略网络
            normalizer=normalizer,                          # 传入策略使用的归一化器
            path=export_model_dir,                          # 指定 ONNX 输出目录
            filename=filename,                              # 指定 ONNX 文件名
        )
    return os.path.join(export_model_dir, filename)         # 返回导出后的 ONNX 文件完整路径



# ==================================================================================
# 创建键盘遥控器，并返回每个仿真步更新命令的回调函数
# ==================================================================================
def make_keyboard_controller(env):
    import numpy as np                                      # 将键盘方向映射转换成 NumPy 数组
    import torch                                            # 将键盘生成的命令转换为 PyTorch 张量
    from isaaclab.devices import (
        Se2Keyboard,
        Se2KeyboardCfg,
    )                                                       # 导入 SE(2) 平面运动键盘设备及其配置

    command_term = env.unwrapped.command_manager.get_term("base_velocity")  # 获取速度命令管理器中的 base_velocity 命令项
    keyboard = Se2Keyboard(                                 # 按前进、横移和转向灵敏度初始化键盘设备
        Se2KeyboardCfg(                                     # 配置三类速度命令的键盘灵敏度，前进 0.8、横移 0.4、转向 1.0
            v_x_sensitivity=0.8,
            v_y_sensitivity=0.4,
            omega_z_sensitivity=1.0,
        )
    )
    keyboard._INPUT_KEY_MAPPING.update(                     # 覆盖默认键位映射，明确设置 Lite3 的遥控方向和幅值
        {
            "W": np.asarray([1.0, 0.0, 0.0]) * 0.8,         # 前进
            "S": np.asarray([-1.0, 0.0, 0.0]) * 0.8,        # 后退
            "A": np.asarray([0.0, 1.0, 0.0]) * 0.4,         # 左移
            "D": np.asarray([0.0, -1.0, 0.0]) * 0.4,        # 右移
            "Q": np.asarray([0.0, 0.0, 1.0]),               # 逆时针转向
            "E": np.asarray([0.0, 0.0, -1.0]),              # 顺时针转向
        }
    )

    def apply_keyboard_command() -> None:                   # 每个控制周期读取并应用键盘速度命令
        command = (
            torch.as_tensor(                                # 读取键盘设备当前命令，并转成环境设备上的 float32 张量
                keyboard.advance(),
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
        )
        command_term.vel_command_b[:] = (
            command                                         # 将键盘命令写入命令项，使策略在下一步使用该指令
        )

    return apply_keyboard_command                           # 将命令更新回调交给播放主循环调用



# ==================================================================================
# 播放和 ONNX 导出主流程
# ==================================================================================
def main() -> None:
    import gymnasium as gym                                 # 创建 Gymnasium 环境
    import torch                                            # 在推理阶段使用无梯度上下文
    from rsl_rl.runners import OnPolicyRunner               # 创建 RSL-RL checkpoint 加载器和推理策略
    from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper)     # 将 Isaac Lab 环境包装成 RSL-RL 所需的向量环境接口
    from isaaclab_tasks.utils.parse_cfg import (load_cfg_from_registry)         # 从任务注册表中读取环境和智能体配置
    import quadruped_robot.tasks                            # 导入任务包以触发 Gymnasium 任务注册

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")      # 读取命令行指定任务的环境配置
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")  # 读取对应的 RSL-RL 策略与训练器配置
    env_cfg.scene.num_envs = 1                              # 播放只启动一个环境，避免多环境占用资源并干扰观察
    env_cfg.observations.policy.enable_corruption = (False) # 关闭策略观测噪声，使用干净观测进行推理
    if args_cli.device is not None:                         # 如果指定了设备，则将仿真与训练器都切换到该设备
        env_cfg.sim.device = args_cli.device                # 设置仿真环境使用的设备
        agent_cfg.device = args_cli.device                  # 设置策略推理使用的设备
    stabilize_play_reset(env_cfg)                           # 固定重置姿态并关闭训练中的外力随机化
    configure_keyboard_command(env_cfg)                     # 配置键盘遥控命令，并关闭命令自动重采样
    if args_cli.export_onnx:                                # ONNX 导出时不需要速度命令的可视化标记
        env_cfg.commands.base_velocity.debug_vis = (False)  # 关闭命令可视化，减少导出模式中的界面内容
    configure_viewer(env_cfg)                               # 设置播放窗口相机位置
    checkpoints = [resolve_checkpoint(agent_cfg)]           # 根据参数指定的路径或默认日志目录解析要加载的 checkpoint
    env = RslRlVecEnvWrapper(                               # 创建 Gymnasium 环境并包装为 RSL-RL 兼容接口
        gym.make(args_cli.task, cfg=env_cfg),
        clip_actions=agent_cfg.clip_actions,                # 使用指定任务和刚才调整后的环境配置创建仿真环境
    )
    try:                                                    # 无论播放、导出还是加载过程是否报错，都关闭环境
        runner = OnPolicyRunner(                            # 初始化推理 runner；播放不写训练日志，因此 log_dir 设为 None
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,                        # 传入环境、智能体配置和推理设备
        )
        if (args_cli.export_onnx):                          # ONNX 导出模式逐个加载并导出 checkpoint，然后结束主流程
            for checkpoint in checkpoints:                  # 遍历待导出的模型路径
                print(f"[INFO] Loading checkpoint: {checkpoint}")  # 打印当前正在处理的 checkpoint
                runner.load(checkpoint)                     # 将模型参数加载到 runner
                exported_path = export_policy_to_onnx(runner, checkpoint)   # 导出 ONNX 并获取输出文件路径
                print(f"[INFO] Exported ONNX policy: {exported_path}")      # 显示生成的 ONNX 文件位置
            return                                          # 完成导出后返回，不进入键盘播放流程
        checkpoint = checkpoints[0]                         # 播放模式使用解析出的第一个 checkpoint
        print(f"[INFO] Loading checkpoint: {checkpoint}")   # 打印当前加载的模型文件
        runner.load(checkpoint)                             # 将 checkpoint 参数载入 runner
        policy = runner.get_inference_policy(device=env.unwrapped.device)  # 创建在指定设备上运行的推理策略函数
        apply_keyboard_command = make_keyboard_controller(env)              # 初始化键盘控制器并获取命令更新回调
        print("[INFO] Keyboard UI mode: use mouse for view, hold W/S/A/D/Q/E or arrow/Z/X keys, press L to stop.")
        obs = env.get_observations()                        # 读取环境的初始策略观测
        obs = (obs[0] if isinstance(obs, tuple) else obs)   # 某些环境接口会返回 (观测, 额外信息)，此处只保留观测张量
        while (simulation_app.is_running()):                # 仿真窗口运行期间持续读取键盘、执行策略并推进仿真
            with torch.inference_mode():                    # 推理过程不计算梯度，降低显存和计算开销
                apply_keyboard_command()                    # 在执行策略前写入当前键盘命令
                obs, _, _, _ = env.step(policy(obs))        # 用当前策略动作推进一个仿真控制步，并更新下一帧观测
                apply_keyboard_command()                    # 再次更新命令，避免命令项被环境步进过程重采样覆盖
    finally:
        env.close()                                         # 关闭向量环境及其底层仿真资源



# ==================================================================================
# 直接运行脚本时启动播放或导出；作为模块导入时不启动仿真
# ==================================================================================
if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
