"""使用固定命令矩阵评测运动策略，并导出逐回合和分类汇总指标。"""

from __future__ import annotations

import argparse  # 定义 benchmark 的命令行参数
import csv  # 读写 rollout 和汇总 CSV 文件
import math  # 执行数值有效性检查和均值计算
import os  # 处理脚本路径和环境变量
import random  # 固定 Python 随机数生成器以复现实验
import re  # 从 checkpoint 文件名解析迭代编号
import sys  # 设置本地模块搜索路径
from collections import defaultdict  # 按命令类别聚合 rollout 数据
from dataclasses import dataclass  # 定义固定命令用例的数据结构
from pathlib import Path  # 处理 checkpoint 和结果文件路径
from statistics import mean, stdev  # 计算指标均值和标准差

from isaaclab.app import AppLauncher  # 在正式 benchmark 时启动 Isaac Sim

sys.path.append(os.path.dirname(__file__))  # 允许导入同目录中的命令行参数辅助模块
import cli_args  # 复用训练和播放脚本的 Isaac Sim 默认参数

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 以脚本位置确定 rl_training 项目根目录


parser = argparse.ArgumentParser(  # 创建固定命令评测工具的命令行解析器
    description="Benchmark an Isaac Lab RSL-RL locomotion checkpoint."
)
parser.add_argument("--task", default=None, help="Registered Gymnasium task ID.")  # 指定要评测的任务；只汇总 CSV 时可省略
parser.add_argument(
    "--checkpoint_path", type=str, default=None,
    help="Checkpoint file or run directory; defaults to the newest checkpoint under logs/rsl_rl.",
)
parser.add_argument("--output_dir", type=str, default=None, help="Directory for benchmark CSV files.")  # 指定结果保存目录
parser.add_argument("--trials", type=int, default=3, help="Independent resets per command (default: 3).")  # 每种命令重复的独立回合数
parser.add_argument("--duration_s", type=float, default=8.0, help="Duration of each command rollout in seconds.")  # 每个命令用例的仿真时长
parser.add_argument("--warmup_s", type=float, default=2.0, help="Initial time excluded from steady-state metrics.")  # 稳态指标统计前忽略的预热时长
parser.add_argument("--seed", type=int, default=42, help="Seed used for reproducible simulator resets.")  # 固定评测随机种子
parser.add_argument(
    "--case",
    choices=(
        "stand", "forward", "backward", "left", "right", "ccw", "cw",
        "left_ccw", "right_cw", "forward_ccw", "forward_cw",
    ),
    default=None,
    help="Evaluate one command case; use --trials 1 for a single visible environment.",
)
parser.add_argument(
    "--nominal_conditions", action="store_true",
    help="Disable startup/reset randomization and use nominal dynamics instead of play.py conditions.",
)
parser.add_argument(
    "--hold_window_s", type=float, default=20.0,
    help="After evaluation, command zero velocity and keep the policy controlling a stand for this many seconds (GUI default: 20).",
)
parser.add_argument(
    "--fall_height_m", type=float, default=0.18,
    help="Count a rollout as fallen when base height drops below this value (default: 0.18 m).",
)
parser.add_argument(
    "--fall_tilt_rad", type=float, default=0.8,
    help="Count a rollout as fallen when body tilt exceeds this value (default: 0.8 rad).",
)
parser.add_argument(
    "--zero_actions", action="store_true",
    help="Apply zero joint offsets instead of the policy for a nominal-stance diagnostic.",
)
parser.add_argument(
    "--initial_height_m", type=float, default=None,
    help="Override the robot's initial root height for a pose diagnostic.",
)
parser.add_argument(
    "--initial_hip_y_rad", type=float, default=None,
    help="Override all four HipY initial angles for a pose diagnostic.",
)
parser.add_argument(
    "--initial_knee_rad", type=float, default=None,
    help="Override all four Knee initial angles for a pose diagnostic.",
)
parser.add_argument(
    "--trace_pose", action="store_true",
    help="Print the first rollout's root, foot, joint, and contact state while it settles.",
)
parser.add_argument(
    "--actuator_damping", type=float, default=None,
    help="Override Hip and Knee actuator damping for a nominal stance diagnostic.",
)
parser.add_argument(
    "--actuator_stiffness", type=float, default=None,
    help="Override Hip and Knee actuator stiffness for a nominal stance diagnostic.",
)
parser.add_argument(
    "--solver_velocity_iterations", type=int, default=None,
    help="Override articulation velocity solver iterations for a nominal stance diagnostic.",
)
parser.add_argument(
    "--solver_position_iterations", type=int, default=None,
    help="Override articulation position solver iterations for a nominal stance diagnostic.",
)
parser.add_argument(
    "--actuator_model", choices=("snapshot", "old_dc"), default="snapshot",
    help="Use the current actuator or the former DC motor at the same pose for a diagnostic.",
)
parser.add_argument(
    "--actuator_delay_steps", type=int, default=None,
    help="Override a delayed PD actuator's fixed delay in physics steps.",
)
parser.add_argument(
    "--linear_command_scale", type=float, default=1.0,
    help="Scale fixed vx and vy commands for an in-curriculum diagnostic.",
)
parser.add_argument(
    "--yaw_command_scale", type=float, default=1.0,
    help="Scale fixed yaw commands for an in-curriculum diagnostic.",
)
parser.add_argument(
    "--summarize_only", action="store_true",
    help="Create summary.csv from an existing rollouts.csv without launching Isaac Sim.",
)
parser.add_argument(
    "--rollouts_path", type=str, default=None,
    help="Input rollouts.csv for --summarize_only; defaults to the newest benchmark result.",
)
AppLauncher.add_app_launcher_args(parser)  # 增加 Isaac Sim 的无头运行和设备选择等参数
args_cli = parser.parse_args()  # 解析命令行输入
if not args_cli.summarize_only and args_cli.task is None:  # 正式仿真评测必须指定任务
    parser.error("--task is required unless --summarize_only is specified")  # 参数缺失时显示用法并退出
if args_cli.trials < 1:  # 检查重复次数是否有效
    parser.error("--trials must be at least 1")  # trials 必须大于等于 1
if args_cli.duration_s <= 0.0 or not 0.0 <= args_cli.warmup_s < args_cli.duration_s:  # 检查时长和预热区间
    parser.error("require duration_s > 0 and 0 <= warmup_s < duration_s")  # 预热不能为负，也必须短于总时长
if args_cli.fall_height_m <= 0.0 or not 0.0 < args_cli.fall_tilt_rad < math.pi:  # 检查跌倒判定阈值
    parser.error("require fall_height_m > 0 and 0 < fall_tilt_rad < pi")  # 高度和倾角阈值必须在有效范围
if args_cli.linear_command_scale <= 0.0 or args_cli.yaw_command_scale <= 0.0:
    parser.error("command scales must be positive")
if args_cli.hold_window_s < 0.0:
    parser.error("--hold_window_s must be nonnegative")
if args_cli.initial_height_m is not None and args_cli.initial_height_m <= 0.0:
    parser.error("--initial_height_m must be positive")
if args_cli.actuator_damping is not None and args_cli.actuator_damping < 0.0:
    parser.error("--actuator_damping must be nonnegative")
if args_cli.actuator_stiffness is not None and args_cli.actuator_stiffness < 0.0:
    parser.error("--actuator_stiffness must be nonnegative")
if args_cli.solver_velocity_iterations is not None and args_cli.solver_velocity_iterations < 0:
    parser.error("--solver_velocity_iterations must be nonnegative")
if args_cli.solver_position_iterations is not None and args_cli.solver_position_iterations < 1:
    parser.error("--solver_position_iterations must be positive")
if args_cli.actuator_delay_steps is not None and args_cli.actuator_delay_steps < 0:
    parser.error("--actuator_delay_steps must be nonnegative")
if args_cli.actuator_model == "old_dc" and args_cli.actuator_delay_steps is not None:
    parser.error("--actuator_delay_steps only applies to snapshot delayed PD actuators")
if not args_cli.summarize_only:  # 汇总已有 CSV 时不需要启动仿真参数默认值
    cli_args.apply_local_app_defaults(args_cli)  # 设置 GPU、渲染和 Kit 参数等本地默认值

simulation_app = None  # 汇总模式不创建 Isaac Sim，因此应用句柄初始为空

LITE3_FOOT_NAMES = ("FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT")  # 接触和步态指标统一使用的足端顺序
LITE3_SHANK_NAMES = ("FL_SHANK", "FR_SHANK", "HL_SHANK", "HR_SHANK")  # 记录非足端承重位置


@dataclass(frozen=True)
class CommandCase:  # 固定评测中的一个速度命令用例
    name: str  # 用于 CSV 和终端输出的用例名
    category: str  # 汇总统计使用的命令类别
    vx: float  # 机身前后方向目标速度，单位 m/s
    vy: float  # 机身横向目标速度，单位 m/s
    wz: float  # 机身偏航角速度目标，单位 rad/s


# 固定命令覆盖站立、前后、横移、原地转向和混合运动，避免策略只适应单一方向。
COMMAND_CASES = (
    CommandCase("stand", "stand", 0.0, 0.0, 0.0),  # 零速度站立
    CommandCase("forward", "forward", 0.6, 0.0, 0.0),  # 正向行走
    CommandCase("backward", "forward", -0.6, 0.0, 0.0),  # 反向行走，与前进归入同一类别
    CommandCase("left", "lateral", 0.0, 0.4, 0.0),  # 向左横移
    CommandCase("right", "lateral", 0.0, -0.4, 0.0),  # 向右横移
    CommandCase("ccw", "yaw", 0.0, 0.0, 1.0),  # 逆时针原地转向
    CommandCase("cw", "yaw", 0.0, 0.0, -1.0),  # 顺时针原地转向
    CommandCase("left_ccw", "lateral_yaw", 0.0, 0.3, 0.7),  # 左移并逆时针转向
    CommandCase("right_cw", "lateral_yaw", 0.0, -0.3, -0.7),  # 右移并顺时针转向
    CommandCase("forward_ccw", "forward_yaw", 0.5, 0.0, 0.7),  # 前进并逆时针转向
    CommandCase("forward_cw", "forward_yaw", 0.5, 0.0, -0.7),  # 前进并顺时针转向
)


def disable_randomization(env_cfg) -> None:  # 为所有评测回合固定名义动力学和初始状态
    """关闭域随机化，使不同 checkpoint 在相同物理条件下比较。"""
    # 关闭会改变机器人或地面物理属性的随机化事件。
    for name in (
        "randomize_rigid_body_material",  # 关闭地面摩擦和恢复系数随机化
        "randomize_rigid_body_mass_base",  # 关闭机身质量随机化
        "randomize_rigid_body_mass_others",  # 关闭其他刚体质量随机化
        "randomize_com_positions",  # 关闭质心偏移随机化
        "randomize_apply_external_force_torque",  # 关闭随机外力与外力矩
        "randomize_actuator_gains",  # 关闭执行器增益随机化
        "randomize_push_robot",  # 关闭随机推动
    ):
        # 仅在对应事件存在于当前任务配置时禁用它。
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    # 固定机器人重置时的位置、姿态和速度扰动。
    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {  # 每个分量都不施加随机偏移
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {  # 每个线速度和角速度分量都从 0 开始
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

    env_cfg.commands.base_velocity.debug_vis = False  # 关闭命令标记，减少评测界面开销
    env_cfg.commands.base_velocity.heading_command = False  # 使用直接偏航角速度命令
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0  # 不采样朝向控制命令
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0  # 固定命令矩阵中的站立样本比例
    env_cfg.commands.base_velocity.resampling_time_range = (1_000_000.0, 1_000_000.0)  # 防止训练命令重采样覆盖 benchmark 命令
    # 关闭训练回合时长终止，避免评测时自动重置用例。
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    # 若任务配置有地形边界终止，也在平地固定评测中关闭它。
    if hasattr(env_cfg.terminations, "terrain_out_of_bounds"):
        env_cfg.terminations.terrain_out_of_bounds = None


def match_play_conditions(env_cfg) -> None:  # 复用 play.py 的重置与随机化设置
    """Match play.py: fixed base reset, but retain startup and actuator randomization."""
    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
    for name in ("randomize_apply_external_force_torque", "randomize_push_robot"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.commands.base_velocity.debug_vis = not args_cli.headless
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1_000_000.0, 1_000_000.0)
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None


def resolve_input_path(value: str) -> Path:  # 解析 checkpoint 或 CSV 的输入路径
    """按当前工作目录或 rl_training 根目录解析输入路径。"""
    path = Path(value).expanduser()  # 展开用户目录简写并构造路径对象
    if path.is_absolute():  # 绝对路径直接规范化
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()  # 首先尝试相对于当前工作目录查找
    if cwd_path.exists():  # 当前目录下存在该路径时优先使用
        return cwd_path
    return (PROJECT_ROOT / path).resolve()  # 否则将相对路径解释为相对于项目根目录


def resolve_output_path(value: str) -> Path:  # 将相对输出目录固定到项目根目录下
    """将相对结果目录固定解析到项目根目录。"""
    path = Path(value).expanduser()  # 展开用户目录简写
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()  # 保留绝对路径，否则拼接项目根目录


def checkpoint_iteration(path: Path) -> int:  # 从模型文件名提取训练迭代编号
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)  # 只匹配 model_<迭代号>.pt 格式
    return int(match.group(1)) if match else -1  # 非标准文件名返回 -1


def resolve_checkpoint(agent_cfg) -> str:  # 根据用户输入或实验目录找到待评测模型
    # 用户指定 checkpoint 文件或运行目录时，优先使用该路径。
    if args_cli.checkpoint_path is not None:
        checkpoint_path = resolve_input_path(args_cli.checkpoint_path)  # 解析输入路径
        if checkpoint_path.is_dir():  # 输入是目录时，从中选迭代号最高的模型
            checkpoints = sorted(
                checkpoint_path.glob("model_*.pt"),
                key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime),
            )
            if not checkpoints:  # 目录下没有模型文件时终止并报告路径
                raise FileNotFoundError(
                    f"No model_*.pt checkpoint found in: {checkpoint_path}"
                )
            return str(checkpoints[-1])  # 返回排序后的最后一个模型文件
        if not checkpoint_path.is_file():  # 输入不是文件或目录时报告错误
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return str(checkpoint_path)  # 输入是有效模型文件时直接返回

    # 未指定路径时，在当前任务对应的实验日志目录中搜索模型。
    log_root = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    checkpoints = sorted(
        log_root.rglob("model_*.pt"),
        key=lambda path: (path.stat().st_mtime, checkpoint_iteration(path)),
    )
    if not checkpoints:  # 搜索结果为空时提示用户指定有效模型
        raise FileNotFoundError(f"No model_*.pt checkpoint found under: {log_root}")
    return str(checkpoints[-1])  # 默认选择最近保存的 checkpoint


def resolve_rollouts_path() -> Path:  # 查找需要重新汇总的 rollouts.csv
    """查找指定的 rollouts.csv，或自动选择最新的评测结果。"""
    if args_cli.rollouts_path is not None:  # 优先采用用户直接指定的 CSV 路径
        path = resolve_input_path(args_cli.rollouts_path)  # 解析输入路径
        if not path.is_file():  # 文件不存在时给出明确错误
            raise FileNotFoundError(f"Rollout metrics do not exist: {path}")
        return path
    if args_cli.output_dir is not None:  # 未指定 CSV 时可从输出目录中读取
        path = resolve_output_path(args_cli.output_dir) / "rollouts.csv"
        if not path.is_file():  # 输出目录尚无原始数据时报告错误
            raise FileNotFoundError(f"Rollout metrics do not exist: {path}")
        return path
    # 默认查找日志树中最新生成的 benchmark/rollouts.csv。
    candidates = list(
        (PROJECT_ROOT / "logs" / "rsl_rl").glob("*/*/benchmark/rollouts.csv")
    )
    if not candidates:  # 没有历史评测文件时提示显式指定路径
        raise FileNotFoundError(
            "No benchmark/rollouts.csv found under logs/rsl_rl; pass --rollouts_path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()  # 返回最近修改的评测文件


def unwrap_observation(observation):  # 兼容不同版本包装器返回的观测格式
    """从包装器返回值中提取策略观测。"""
    return observation[0] if isinstance(observation, tuple) else observation  # 返回纯观测张量


def build_foot_indices(robot, contact_sensor):  # 按步态时钟的顺序查找四足刚体和传感器索引
    """按 FL、FR、HL、HR 顺序查找足端索引。"""
    asset_ids, asset_names = robot.find_bodies(
        list(LITE3_FOOT_NAMES), preserve_order=True
    )  # 获取机器人资产中的足端索引
    sensor_ids, sensor_names = contact_sensor.find_bodies(
        list(LITE3_FOOT_NAMES), preserve_order=True
    )  # 获取接触传感器中对应足端的索引
    if (
        tuple(asset_names) != LITE3_FOOT_NAMES
        or tuple(sensor_names) != LITE3_FOOT_NAMES
    ):
        raise RuntimeError(
            "Could not resolve all Lite3 feet in FL/FR/HL/HR order: "
            f"asset={asset_names}, sensor={sensor_names}"
        )
    return list(LITE3_FOOT_NAMES), asset_ids, sensor_ids  # 返回足端名称及两组索引


def evaluate_batch(env, policy, runs, foot_names, asset_foot_ids, sensor_foot_ids):  # 并行运行全部命令和试验组合
    """并行评测所有命令和 trial，确保用例使用相同仿真条件。"""
    import torch  # 在 Isaac Sim 启动后导入 PyTorch

    # 初始化所有并行环境，并取得底层环境和机器人数据接口。
    obs = unwrap_observation(env.reset())
    unwrapped = env.unwrapped  # 访问未包装的 Isaac Lab 环境
    robot = unwrapped.scene["robot"]  # 读取机器人关节和刚体状态
    hipx_ids, _ = robot.find_joints(".*_HipX_joint", preserve_order=True)
    hipy_ids, _ = robot.find_joints(".*_HipY_joint", preserve_order=True)
    knee_ids, _ = robot.find_joints(".*_Knee_joint", preserve_order=True)
    contact_sensor = unwrapped.scene.sensors["contact_forces"]  # 读取足端接触数据
    shank_sensor_ids, shank_names = contact_sensor.find_bodies(list(LITE3_SHANK_NAMES), preserve_order=True)
    if tuple(shank_names) != LITE3_SHANK_NAMES:
        raise RuntimeError(f"Could not resolve Lite3 shanks in FL/FR/HL/HR order: {shank_names}")
    command_term = unwrapped.command_manager.get_term("base_velocity")  # 获取速度命令项
    # 将每个 trial 对应的三轴命令放到 GPU 张量中并写入环境。
    commands = torch.tensor(
        [[case.vx, case.vy, case.wz] for _, case in runs],
        device=unwrapped.device,
        dtype=torch.float32,
    )
    command_term.vel_command_b[:] = commands  # 覆盖重置时随机采样的速度命令
    obs = env.get_observations()  # 刷新首帧观测，确保策略看到的是固定评测命令

    # 将评测时长和预热时长换算成仿真步数。
    total_steps = round(args_cli.duration_s / unwrapped.step_dt)
    warmup_steps = round(args_cli.warmup_s / unwrapped.step_dt)
    num_envs = len(runs)  # 每个命令和 trial 组合占用一个并行环境
    # 初始化位移统计，用于计算稳态行走速度和转向漂移。
    measurement_start_position = torch.zeros(num_envs, 2, device=unwrapped.device)
    measurement_last_position = torch.zeros_like(measurement_start_position)
    measurement_started = torch.zeros(
        num_envs, dtype=torch.bool, device=unwrapped.device
    )
    sample_count = torch.zeros(num_envs, device=unwrapped.device)  # 每个环境有效的稳态采样数
    gait_sample_count = torch.zeros(num_envs, device=unwrapped.device)  # 每个运动环境的步态采样数
    moving_command = torch.linalg.vector_norm(commands, dim=1) > 1.0e-4  # 排除站立命令的步态统计
    previous_action = None
    previous_lin_vel_z = None
    # 这些指标只对运动命令累计，不将站立结果混入步态质量。
    gait_metric_names = {
        "gait_clock_match",
        "diag_sync_fl_hr",
        "diag_sync_fr_hl",
        "diag_antiphase",
    }
    # 为速度误差、姿态、动作平滑度、功率和步态等指标创建累加器。
    samples = {
        name: torch.zeros(num_envs, device=unwrapped.device)
        for name in (
            "mae_vx",
            "mae_vy",
            "mae_wz",
            "actual_vx",
            "actual_vy",
            "actual_wz",
            "vertical_speed_abs",
            "tilt_rad",
            "roll_abs_rad",
            "pitch_abs_rad",
            "base_height_m",
            "vertical_acc_abs",
            "action_rate_l2",
            "joint_deviation_mean_abs_rad",
            "hipx_deviation_mean_abs_rad",
            "hipy_deviation_mean_abs_rad",
            "knee_deviation_mean_abs_rad",
            "joint_power_w",
            "gait_clock_match",
            "diag_sync_fl_hr",
            "diag_sync_fr_hl",
            "diag_antiphase",
        )
    }
    # 初始化足端滑移、摆动高度、刮地率和支撑占空比统计量。
    stance_slip_sum = torch.zeros(num_envs, device=unwrapped.device)
    stance_contact_count = torch.zeros(num_envs, device=unwrapped.device)
    swing_height_sum = torch.zeros(num_envs, device=unwrapped.device)
    swing_contact_sum = torch.zeros(num_envs, device=unwrapped.device)
    swing_sample_count = torch.zeros(num_envs, device=unwrapped.device)
    duty_sum = torch.zeros(num_envs, len(foot_names), device=unwrapped.device)
    shank_duty_sum = torch.zeros(num_envs, len(shank_names), device=unwrapped.device)
    # 记录终止状态、跌倒状态和首次跌倒时的诊断信息。
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    fall_detected = torch.zeros_like(terminated)
    first_fall_step = torch.full((num_envs,), -1, dtype=torch.int, device=unwrapped.device)
    first_fall_height = torch.full((num_envs,), float("nan"), device=unwrapped.device)
    first_fall_tilt = torch.full_like(first_fall_height, float("nan"))
    first_fall_by_height = torch.zeros_like(terminated)
    first_fall_by_tilt = torch.zeros_like(terminated)
    reset_height = robot.data.root_pos_w[:, 2].clone()  # 保存初始机身高度供结果表记录
    reset_tilt = torch.acos(
        torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
    )  # 根据重力投影计算初始倾角
    if args_cli.trace_pose:
        print("[POSE] joint_names=" + str(robot.joint_names), flush=True)
        undesired_cfg = unwrapped.reward_manager.get_term_cfg("undesired_contacts")
        print(
            "[POSE] undesired_contact_bodies="
            + str(undesired_cfg.params["sensor_cfg"].body_names)
            + " ids="
            + str(undesired_cfg.params["sensor_cfg"].body_ids),
            flush=True,
        )
        print(
            f"[POSE] t=0.00 root_z={float(reset_height[0]):.4f} "
            f"foot_z={robot.data.body_pos_w[0, asset_foot_ids, 2].tolist()} "
            f"joint_q={robot.data.joint_pos[0].tolist()}", flush=True,
        )

    # 同时推进全部环境，遇到跌倒后停止累计该环境的指标。
    for step in range(total_steps):
        command_term.vel_command_b[:] = commands  # 每步重新写入固定命令，防止环境内部覆盖
        with torch.inference_mode():
            # 零动作模式用于区分初始姿态稳定性与策略控制效果。
            actions = (
                torch.zeros_like(unwrapped.action_manager.action)
                if args_cli.zero_actions
                else policy(obs)
            )
            obs, _, dones, _ = env.step(actions)  # 执行策略动作并推进一个控制周期
        command_term.vel_command_b[:] = commands  # 恢复固定命令供下一步观测使用
        obs = unwrap_observation(obs)  # 兼容包装器返回的观测结构
        dones = torch.as_tensor(dones, device=unwrapped.device, dtype=torch.bool)  # 统一终止标志的数据类型和设备
        gravity_b = robot.data.projected_gravity_b  # 获取机身坐标系下的重力投影
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
        # ManagerBasedRLEnv auto-resets terminated environments inside env.step().
        # The post-step root pose for those environments is already the reset pose,
        # so read the termination manager's per-term flags before classifying falls.
        termination_manager = unwrapped.termination_manager
        termination_flags = {
            name: termination_manager.get_term(name).clone()
            for name in termination_manager.active_terms
        }
        manager_fell_height = termination_flags.get(
            "fallen_height", torch.zeros_like(dones)
        ).bool()
        manager_fell_tilt = termination_flags.get(
            "fallen_orientation", torch.zeros_like(dones)
        ).bool()
        low_height = robot.data.root_pos_w[:, 2] < args_cli.fall_height_m
        high_tilt = tilt > args_cli.fall_tilt_rad
        fell = manager_fell_height | manager_fell_tilt | ((low_height | high_tilt) & ~dones)
        if args_cli.trace_pose and step in (0, 4, 9, 19, 29, 39, 49, 59, 79, 99):
            all_contact_force = contact_sensor.data.net_forces_w_history[0].norm(dim=-1).max(dim=0)[0]
            trace_contact_force = all_contact_force[sensor_foot_ids]
            undesired_cfg = unwrapped.reward_manager.get_term_cfg("undesired_contacts")
            undesired_count = undesired_cfg.func(unwrapped, **undesired_cfg.params)[0]
            undesired_term_id = unwrapped.reward_manager._term_names.index("undesired_contacts")
            undesired_reward = unwrapped.reward_manager._step_reward[0, undesired_term_id]
            loaded_bodies = [
                (name, round(float(force), 2))
                for name, force in zip(contact_sensor.body_names, all_contact_force)
                if float(force) > 1.0
            ]
            print(
                f"[POSE] t={(step + 1) * unwrapped.step_dt:.2f} "
                f"root_z={float(robot.data.root_pos_w[0, 2]):.4f} "
                f"root_quat={robot.data.root_quat_w[0].tolist()} "
                f"tilt={float(tilt[0]):.3f} "
                f"foot_xy={robot.data.body_pos_w[0, asset_foot_ids, :2].tolist()} "
                f"foot_z={robot.data.body_pos_w[0, asset_foot_ids, 2].tolist()} "
                f"contact_force={trace_contact_force.tolist()} "
                f"contact={(trace_contact_force > 1.0).tolist()} "
                f"undesired_count={float(undesired_count):.1f} "
                f"undesired_reward={float(undesired_reward):.1f} "
                f"loaded_bodies={loaded_bodies} "
                f"action={unwrapped.action_manager.action[0].tolist()} "
                f"joint_q={robot.data.joint_pos[0].tolist()} "
                f"torque={robot.data.applied_torque[0].tolist()}", flush=True,
            )
        first_fall = fell & ~terminated
        first_fall_step = torch.where(first_fall, step + 1, first_fall_step)  # 仅记录首次跌倒步数
        first_fall_height = torch.where(
            first_fall & manager_fell_height,
            torch.full_like(first_fall_height, args_cli.fall_height_m),
            torch.where(first_fall & ~dones, robot.data.root_pos_w[:, 2], first_fall_height),
        )
        first_fall_tilt = torch.where(
            first_fall & manager_fell_tilt,
            torch.full_like(first_fall_tilt, args_cli.fall_tilt_rad),
            torch.where(first_fall & ~dones, tilt, first_fall_tilt),
        )
        first_fall_by_height |= first_fall & (manager_fell_height | (low_height & ~dones))
        first_fall_by_tilt |= first_fall & (manager_fell_tilt | (high_tilt & ~dones))
        fall_detected |= first_fall
        valid = ~(terminated | dones | fell)  # 排除终止、跌倒以及此前已结束的环境
        terminated |= dones | fell  # 更新该 rollout 的结束状态

        # 预热阶段不计入稳态指标，但保留上一帧数据供后续差分计算。
        if step < warmup_steps:
            previous_action = unwrapped.action_manager.action.clone()
            previous_lin_vel_z = robot.data.root_lin_vel_b[:, 2].clone()
            continue

        # 记录预热结束后的水平位移，并为有效环境累计样本数。
        position = robot.data.root_pos_w[:, :2]
        first_valid = valid & ~measurement_started
        measurement_start_position[first_valid] = position[first_valid]
        measurement_last_position[valid] = position[valid]
        measurement_started |= valid
        valid_float = valid.float()
        gait_valid = valid & moving_command
        gait_valid_float = gait_valid.float()
        sample_count += valid_float
        gait_sample_count += gait_valid_float

        # 计算机身实际速度与命令之间的误差。
        actual = robot.data.root_lin_vel_b
        angular = robot.data.root_ang_vel_b
        command_error = commands - torch.stack(
            (actual[:, 0], actual[:, 1], angular[:, 2]), dim=1
        )
        # 对有效稳态样本累计速度误差、实际速度和机身高度。
        samples["mae_vx"] += command_error[:, 0].abs() * valid_float
        samples["mae_vy"] += command_error[:, 1].abs() * valid_float
        samples["mae_wz"] += command_error[:, 2].abs() * valid_float
        samples["actual_vx"] += actual[:, 0] * valid_float
        samples["actual_vy"] += actual[:, 1] * valid_float
        samples["actual_wz"] += angular[:, 2] * valid_float
        samples["vertical_speed_abs"] += actual[:, 2].abs() * valid_float
        samples["base_height_m"] += robot.data.root_pos_w[:, 2] * valid_float

        # 记录倾角，并用相邻步的垂直速度差估算垂直加速度。
        samples["tilt_rad"] += tilt * valid_float
        # 利用机身坐标系重力投影分别估算横滚和俯仰角，避免只看合成倾角。
        roll = torch.atan2(-gravity_b[:, 1], -gravity_b[:, 2])
        pitch = torch.asin(torch.clamp(gravity_b[:, 0], -1.0, 1.0))
        samples["roll_abs_rad"] += roll.abs() * valid_float
        samples["pitch_abs_rad"] += pitch.abs() * valid_float
        if previous_lin_vel_z is not None:
            vertical_acc = (actual[:, 2] - previous_lin_vel_z) / unwrapped.step_dt
            samples["vertical_acc_abs"] += vertical_acc.abs() * valid_float
        previous_lin_vel_z = actual[:, 2].clone()

        # 根据接触力历史判断每只脚是否着地，并计算支撑脚滑移速度。
        contact = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
            > 1.0
        )
        foot_vel_xy_w = robot.data.body_lin_vel_w[:, asset_foot_ids, :2].norm(dim=-1)
        stance_slip_sum += (foot_vel_xy_w * contact).sum(dim=1) * valid_float
        stance_contact_count += contact.sum(dim=1) * valid_float
        duty_sum += contact.float() * valid_float.unsqueeze(1)
        shank_contact = (
            contact_sensor.data.net_forces_w_history[:, :, shank_sensor_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
            > 1.0
        )
        shank_duty_sum += shank_contact.float() * valid_float.unsqueeze(1)

        # 按 0.6 秒周期和对角相位偏移生成预期支撑相，评估时钟匹配度。
        phase = (unwrapped.episode_length_buf * unwrapped.step_dt / 0.6).unsqueeze(
            1
        ) % 1.0
        offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=unwrapped.device)
        expected_stance = ((phase + offsets) % 1.0) < 0.56
        samples["gait_clock_match"] += (expected_stance == contact).float().mean(
            dim=1
        ) * gait_valid_float
        samples["diag_sync_fl_hr"] += (
            contact[:, 0] == contact[:, 3]
        ).float() * gait_valid_float
        samples["diag_sync_fr_hl"] += (
            contact[:, 1] == contact[:, 2]
        ).float() * gait_valid_float
        group_a_stance = contact[:, 0] & contact[:, 3]
        group_b_stance = contact[:, 1] & contact[:, 2]
        samples["diag_antiphase"] += (
            torch.logical_xor(group_a_stance, group_b_stance).float() * gait_valid_float
        )

        # 只在预期摆动相统计足端高度和触地比例，后者作为刮地率。
        expected_swing = ~expected_stance
        valid_swing = expected_swing & gait_valid.unsqueeze(1)
        swing_height_sum += (
            robot.data.body_pos_w[:, asset_foot_ids, 2] * valid_swing
        ).sum(dim=1)
        swing_contact_sum += (contact & valid_swing).sum(dim=1)
        swing_sample_count += valid_swing.sum(dim=1)

        # 累计动作变化、关节偏离默认角和关节功率。
        action = unwrapped.action_manager.action
        if previous_action is not None:
            samples["action_rate_l2"] += (
                torch.linalg.vector_norm(action - previous_action, dim=1) * valid_float
            )
        previous_action = action.clone()
        samples["joint_deviation_mean_abs_rad"] += (
            torch.abs(robot.data.joint_pos - robot.data.default_joint_pos).mean(dim=1)
            * valid_float
        )
        joint_deviation = torch.abs(robot.data.joint_pos - robot.data.default_joint_pos)
        samples["hipx_deviation_mean_abs_rad"] += joint_deviation[:, hipx_ids].mean(dim=1) * valid_float
        samples["hipy_deviation_mean_abs_rad"] += joint_deviation[:, hipy_ids].mean(dim=1) * valid_float
        samples["knee_deviation_mean_abs_rad"] += joint_deviation[:, knee_ids].mean(dim=1) * valid_float
        samples["joint_power_w"] += (
            torch.abs(robot.data.joint_vel * robot.data.applied_torque).sum(dim=1)
            * valid_float
    )

    # 对没有任何有效样本的指标返回 NaN，避免产生除零结果。
    def divide_or_nan(numerator, denominator):
        denominator = denominator.expand_as(numerator)  # 将分母扩展到与分子相同的形状
        return torch.where(
            denominator > 0,
            numerator / torch.clamp(denominator, min=1.0),
            torch.full_like(numerator, float("nan")),
        )  # 有样本时计算均值，无样本时保留 NaN

    # 按每个指标适用的样本数计算均值。
    samples = {
        name: divide_or_nan(
            values, gait_sample_count if name in gait_metric_names else sample_count
        )
        for name, values in samples.items()
    }
    stance_slip = divide_or_nan(stance_slip_sum, stance_contact_count)  # 计算支撑脚平均滑移速度
    swing_height = divide_or_nan(swing_height_sum, swing_sample_count)  # 计算预期摆动相的平均足端高度
    swing_scuff_rate = divide_or_nan(swing_contact_sum, swing_sample_count)  # 计算摆动相触地比例
    duty = divide_or_nan(duty_sum, sample_count.unsqueeze(1))  # 计算每只脚的接触占空比
    shank_duty = divide_or_nan(shank_duty_sum, sample_count.unsqueeze(1))  # 计算小腿接触占空比
    measured_time = sample_count * unwrapped.step_dt  # 将有效样本数换算为实际统计时长
    # 根据预热结束后的起止位置计算水平位移速度。
    displacement = divide_or_nan(
        torch.linalg.vector_norm(
            measurement_last_position - measurement_start_position, dim=1
        ),
        measured_time,
    )

    # 将 GPU 张量中的结果整理为逐 rollout 的 CSV 行。
    rows = []
    for index, (trial, case) in enumerate(runs):
        # 同时保留命令、跌倒诊断、运动质量和姿态指标。
        row = {
            "case": case.name,
            "category": case.category,
            "cmd_vx": case.vx,
            "cmd_vy": case.vy,
            "cmd_wz": case.wz,
            "trial": trial,
            "terminated": int(terminated[index].item()),
            "fall_detected": int(fall_detected[index].item()),
            "fall_step": int(first_fall_step[index].item())
            if first_fall_step[index] > 0
            else float("nan"),
            "fall_elapsed_s": float(first_fall_step[index] * unwrapped.step_dt)
            if first_fall_step[index] > 0
            else float("nan"),
            "fall_by_height": int(first_fall_by_height[index].item()),
            "fall_by_tilt": int(first_fall_by_tilt[index].item()),
            "fall_base_height_m": float(first_fall_height[index]),
            "fall_tilt_rad": float(first_fall_tilt[index]),
            "reset_base_height_m": float(reset_height[index]),
            "reset_tilt_rad": float(reset_tilt[index]),
            "world_displacement_mps": float(displacement[index]),
            "yaw_translation_drift_mps": float(displacement[index])
            if case.category == "yaw"
            else float("nan"),
            "measured_duration_s": float(measured_time[index]),
            "stance_slip_mps": float(stance_slip[index]),
            "swing_height_m": float(swing_height[index]),
            "swing_scuff_rate": float(swing_scuff_rate[index]),
        }
        row.update({name: float(values[index]) for name, values in samples.items()})  # 写入速度、姿态和步态均值
        row.update(
            {
                f"duty_{name}": float(duty[index, foot_index])
                for foot_index, name in enumerate(foot_names)
            }
        )  # 将每只脚的占空比作为独立字段保存
        row.update(
            {
                f"duty_{name}": float(shank_duty[index, shank_index])
                for shank_index, name in enumerate(shank_names)
            }
        )
        rows.append(row)  # 添加当前命令和 trial 的结果
    return rows  # 返回全部 rollout 结果


def hold_zero_command_stand(env, policy, duration_s):  # 撤掉行走命令后继续用策略维持站立
    """将速度命令归零，并在 GUI 保持期间持续运行策略闭环。"""
    import torch

    unwrapped = env.unwrapped
    command_term = unwrapped.command_manager.get_term("base_velocity")
    command_term.vel_command_b.zero_()
    num_steps = round(duration_s / unwrapped.step_dt)
    reset_count = 0
    for _ in range(num_steps):
        if not simulation_app.is_running():
            break
        command_term.vel_command_b.zero_()
        obs = unwrap_observation(env.get_observations())
        with torch.inference_mode():
            actions = (
                torch.zeros_like(unwrapped.action_manager.action)
                if args_cli.zero_actions
                else policy(obs)
            )
            _, _, dones, _ = env.step(actions)
        reset_count += int(torch.as_tensor(dones).sum().item())
        # env.step() 可能自动重置已终止环境并重采样命令；再次清零后刷新观测，
        # 确保下一次策略推理看到的仍是零速度命令。
        command_term.vel_command_b.zero_()
        obs = unwrap_observation(env.get_observations())
    robot = unwrapped.scene["robot"]
    final_tilt = torch.acos(
        torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
    )
    print(
        f"[BENCHMARK] zero-command hold complete: base_z={robot.data.root_pos_w[:, 2].tolist()}, "
        f"tilt_rad={final_tilt.tolist()}, auto_resets={reset_count}",
        flush=True,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:  # 将结果行写入 CSV 文件
    fieldnames = sorted({field for row in rows for field in row})  # 汇总字段名并排序，保证列顺序稳定
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)  # 创建字典到 CSV 行的写入器
        writer.writeheader()  # 写入 CSV 列名
        writer.writerows(rows)  # 写入所有 rollout 或汇总数据


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:  # 按命令类别计算指标均值和标准差
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)  # 为每类命令收集 rollout 行
    for row in rows:
        grouped[str(row["category"])].append(row)  # 使用 category 字段分组
    summary = []
    # 对每个类别分别统计数值列，忽略 NaN 等无效值。
    for category, group in sorted(grouped.items()):
        result: dict[str, object] = {"category": category, "rollouts": len(group)}  # 记录类别名和样本数
        numeric_fields = [key for key in group[0] if key not in {"case", "category"}]  # 排除文本字段
        for field in numeric_fields:
            # 只保留有限的数值，跳过空值、字符串和 NaN。
            values = [
                float(row[field])
                for row in group
                if isinstance(row.get(field), (int, float))
                and math.isfinite(float(row[field]))
            ]
            if values:
                result[f"{field}_mean"] = mean(values)  # 记录该类别的指标均值
                result[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0  # 单样本标准差记为 0
        summary.append(result)  # 添加当前类别的汇总结果
    return summary  # 返回所有类别的统计行


def summarize_rollouts_file(rollouts_path: Path) -> Path:  # 从已有原始 CSV 生成类别汇总
    """不启动 Isaac Sim，直接从原始 CSV 生成分类汇总。"""
    with rollouts_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # 按列名读取原始 rollout 数据
        rows = []
        for source_row in reader:
            case = source_row.pop("case", None)  # 取出命令用例名
            category = source_row.pop("category", None)  # 取出类别名
            if case is None or category is None:
                raise ValueError(f"Missing case/category columns in: {rollouts_path}")
            row = {"case": case, "category": category}  # 保留两个文本字段
            # 将剩余字段转为数字；空字段统一转换为 NaN。
            row.update(
                {
                    name: float(value) if value not in (None, "") else float("nan")
                    for name, value in source_row.items()
                }
            )
            rows.append(row)  # 保存转换后的 rollout 行
    if not rows:  # 空 CSV 无法生成类别汇总
        raise ValueError(f"No rollout rows found in: {rollouts_path}")
    summary_path = rollouts_path.with_name("summary.csv")  # 将汇总文件放在原始 CSV 同目录
    write_csv(summary_path, summarize(rows))  # 写入按类别计算的统计数据
    return summary_path  # 返回汇总文件路径


def main() -> None:  # 启动仿真并运行固定命令评测
    import gymnasium as gym  # 创建已注册的任务环境
    import torch  # 固定 PyTorch 随机种子并运行策略推理
    from rsl_rl.runners import OnPolicyRunner  # 加载 RSL-RL 模型并生成推理策略

    from isaaclab.actuators import DCMotorCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # 将 Isaac Lab 环境适配到 RSL-RL 接口
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # 读取任务注册的环境和策略配置

    import quadruped_robot.tasks  # noqa: F401  # 导入任务包以执行 Gymnasium 注册

    random.seed(args_cli.seed)  # 固定 Python 随机源
    torch.manual_seed(args_cli.seed)  # 固定 PyTorch 随机源
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")  # 读取仿真环境配置
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")  # 读取训练器和模型配置
    if args_cli.actuator_model == "old_dc":
        env_cfg.scene.robot.actuators = {
            "Hip": DCMotorCfg(
                joint_names_expr=[".*_Hip[X,Y]_joint"],
                effort_limit=24.0, saturation_effort=24.0, velocity_limit=26.2,
                stiffness=30.0, damping=0.5, friction=0.0, armature=0.0,
            ),
            "Knee": DCMotorCfg(
                joint_names_expr=[".*_Knee_joint"],
                effort_limit=36.0, saturation_effort=36.0, velocity_limit=17.3,
                stiffness=30.0, damping=0.5, friction=0.0, armature=0.0,
            ),
        }
    if args_cli.initial_height_m is not None:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.initial_height_m)
    if args_cli.initial_hip_y_rad is not None:
        env_cfg.scene.robot.init_state.joint_pos[".*HipY_joint"] = args_cli.initial_hip_y_rad
    if args_cli.initial_knee_rad is not None:
        env_cfg.scene.robot.init_state.joint_pos[".*Knee_joint"] = args_cli.initial_knee_rad
    if args_cli.actuator_damping is not None:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.damping = args_cli.actuator_damping
    if args_cli.actuator_stiffness is not None:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.stiffness = args_cli.actuator_stiffness
    if args_cli.solver_velocity_iterations is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = (
            args_cli.solver_velocity_iterations
        )
    if args_cli.solver_position_iterations is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = (
            args_cli.solver_position_iterations
        )
    if args_cli.actuator_delay_steps is not None:
        for actuator in env_cfg.scene.robot.actuators.values():
            actuator.min_delay = args_cli.actuator_delay_steps
            actuator.max_delay = args_cli.actuator_delay_steps
    command_cases = tuple(
        CommandCase(
            case.name, case.category,
            case.vx * args_cli.linear_command_scale,
            case.vy * args_cli.linear_command_scale,
            case.wz * args_cli.yaw_command_scale,
        )
        for case in COMMAND_CASES
        if args_cli.case is None or case.name == args_cli.case
    )
    runs = [(trial, case) for trial in range(args_cli.trials) for case in command_cases]  # 展开所有 trial 与命令组合
    env_cfg.scene.num_envs = len(runs)  # 每种组合并行占用一个环境
    env_cfg.viewer.env_index = 0  # 可视化时始终让视口跟随第一个评测环境
    env_cfg.viewer.eye = (1.6, -1.6, 0.95)  # 沿用 Play 的观察角度，便于查看机身和四肢
    env_cfg.viewer.lookat = (0.0, 0.0, 0.28)
    env_cfg.viewer.origin_type = "world"
    env_cfg.seed = args_cli.seed  # 将评测随机种子传给仿真环境
    env_cfg.observations.policy.enable_corruption = False  # 关闭观测噪声，使用确定性策略输入
    if args_cli.nominal_conditions:
        disable_randomization(env_cfg)  # 关闭训练阶段的动力学和初始状态随机化
    else:
        match_play_conditions(env_cfg)  # 默认使用与 Play 相同的环境随机化设置
    # 若指定设备，同步设置仿真和策略运行设备。
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device  # 设置仿真设备
        agent_cfg.device = args_cli.device  # 设置策略设备

    checkpoint = resolve_checkpoint(agent_cfg)  # 查找本次要评估的模型文件
    # 默认将结果保存到 checkpoint 旁的 benchmark 目录。
    output_dir = (
        resolve_output_path(args_cli.output_dir)
        if args_cli.output_dir is not None
        else Path(checkpoint).parent / "benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建结果目录
    raw_path = output_dir / "rollouts.csv"  # 设置逐 rollout 结果文件路径
    summary_path = output_dir / "summary.csv"  # 设置类别汇总文件路径
    print(f"[BENCHMARK] loading checkpoint: {checkpoint}")  # 输出当前评测的模型
    print(
        "[BENCHMARK] environment preset: "
        + ("nominal (randomization disabled)" if args_cli.nominal_conditions else "play.py conditions"),
        flush=True,
    )
    if args_cli.zero_actions:
        print("[BENCHMARK] zero-action nominal-stance diagnostic enabled")  # 标明正在运行零动作诊断
    # 创建环境并应用与训练一致的动作裁剪设置。
    env = RslRlVecEnvWrapper(
        gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions
    )
    try:  # 确保评测结束或报错时关闭环境
        print("[BENCHMARK] creating RSL-RL runner")  # 提示正在创建推理器
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        print("[BENCHMARK] restoring policy weights")  # 提示正在加载模型参数
        runner.load(checkpoint)  # 加载指定 checkpoint
        policy = runner.get_inference_policy(device=env.unwrapped.device)  # 构造推理策略
        print("[BENCHMARK] resolving foot contacts")  # 提示正在查找足端和接触传感器索引
        robot = env.unwrapped.scene["robot"]  # 获取机器人资产
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]  # 获取接触力传感器
        foot_names, asset_foot_ids, sensor_foot_ids = build_foot_indices(
            robot, contact_sensor
        )

        print(
            f"[BENCHMARK] evaluating {len(runs)} rollout(s)"
            + (f" for case={args_cli.case}" if args_cli.case else " in parallel"),
            flush=True,
        )
        if not args_cli.headless:
            print(
                "[BENCHMARK] Isaac Sim GUI is enabled; viewport follows environment 0. "
                "Use --case <name> to show one command at a time.",
                flush=True,
            )
        rows = evaluate_batch(
            env, policy, runs, foot_names, asset_foot_ids, sensor_foot_ids
        )
        write_csv(raw_path, rows)  # 写入每个命令和 trial 的原始指标
        write_csv(summary_path, summarize(rows))  # 写入按类别汇总的均值和标准差
        # 在终端简要显示每条 rollout 的三轴速度误差。
        for row in rows:
            print(
                f"[BENCHMARK] trial={row['trial']} case={row['case']} "
                f"mae(vx,vy,wz)=({row['mae_vx']:.3f}, {row['mae_vy']:.3f}, {row['mae_wz']:.3f})",
                flush=True,
            )

        print(f"[BENCHMARK] checkpoint: {checkpoint}")  # 显示评测模型路径
        print(f"[BENCHMARK] rollout metrics: {raw_path}")  # 显示原始结果文件路径
        print(f"[BENCHMARK] category summary: {summary_path}")  # 显示汇总结果文件路径
        if not args_cli.headless and args_cli.hold_window_s > 0.0:
            print(
                f"[BENCHMARK] evaluation commands cleared; policy will hold zero-velocity stand "
                f"for {args_cli.hold_window_s:.1f}s. Close the window to exit early.",
                flush=True,
            )
            hold_zero_command_stand(env, policy, args_cli.hold_window_s)
    finally:
        env.close()  # 关闭环境并释放仿真资源


if __name__ == "__main__":
    if args_cli.summarize_only:  # 只汇总已有 CSV 时不启动 Isaac Sim
        rollouts_path = resolve_rollouts_path()  # 查找输入的 rollouts.csv
        print(f"[BENCHMARK] rollout metrics: {rollouts_path}")  # 输出原始 CSV 路径
        summary_path = summarize_rollouts_file(rollouts_path)  # 生成 summary.csv
        print(f"[BENCHMARK] category summary: {summary_path}")  # 输出汇总 CSV 路径
    else:
        app_launcher = AppLauncher(args_cli)  # 正式评测时启动 Isaac Sim
        simulation_app = app_launcher.app  # 保存仿真应用对象供清理使用
        try:
            main()  # 执行并行 benchmark
        finally:
            simulation_app.close()  # 无论正常退出或异常都关闭 Isaac Sim
