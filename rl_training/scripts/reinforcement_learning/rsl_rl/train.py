"""Train a project task with Isaac Lab 2.x and RSL-RL."""


import argparse                                 # 解析任务、训练参数和 Isaac Sim 启动参数。
import faulthandler                            # 可选地输出长时间无进展时的 Python 调用栈。
import os                                       # 处理当前脚本路径和训练日志目录。
import sys                                      # 将本目录加入模块搜索路径，以便导入本地命令行辅助模块。
from datetime import datetime                   # 用当前时间生成本次训练的独立日志目录名。
from isaaclab.app import AppLauncher            # 启动 Isaac Sim；与仿真相关的模块需在启动后导入。

sys.path.append(os.path.dirname(__file__))      # 将本脚本所在目录加入模块搜索路径。
import cli_args

parser = argparse.ArgumentParser(description="Train an Isaac Lab task with RSL-RL.")    # 创建命令行解析器
parser.add_argument("--task", required=True, help="Registered Gymnasium task ID.")      # 指定已注册的 Gymnasium 任务名称，此参数必填
parser.add_argument("--num_envs", type=int, default=None, help="Override the configured environment count.")    # 可在启动时覆盖环境配置中的并行环境数量
parser.add_argument("--max_iterations", type=int, default=None, help="Override the PPO iteration count.")       # 可在启动时覆盖 PPO 的训练迭代次数
parser.add_argument(
    "--resume_full_command_range",
    action="store_true",
    help="When resuming a trained Lite3 policy, start the new environment with the full velocity-command range.",
)
parser.add_argument(
    "--debug_startup",
    action="store_true",
    help="Dump Python thread stacks if the first vectorized environment step takes more than 30 seconds.",
)

cli_args.add_rsl_rl_args(parser)                # 添加恢复训练、checkpoint、运行名、随机种子等项目参数
AppLauncher.add_app_launcher_args(parser)       # 添加 --headless、--device 等 Isaac Sim 启动参数
args_cli = parser.parse_args()                  # 解析命令行输入
cli_args.apply_local_app_defaults(args_cli)     # 应用本地默认的 Isaac Sim 启动参数，如设备和渲染设置
app_launcher = AppLauncher(args_cli)            # 按命令行参数启动 Isaac Sim
simulation_app = app_launcher.app               # 保存仿真应用实例，供程序结束时关闭



#==================================================================================
# 主函数入口
# 训练主流程在 Isaac Sim 启动后执行
#==================================================================================
def main() -> None:
    import gymnasium as gym                     # 创建已注册的 Gymnasium 环境
    import torch                                # 设置训练所用的 PyTorch 后端选项
    from rsl_rl.runners import OnPolicyRunner   # 使用 RSL-RL 的在线策略训练器
    from isaaclab.envs import ManagerBasedRLEnvCfg                      # 为环境配置提供类型标注
    from isaaclab.utils.io import dump_yaml                             # 将实际使用的环境和智能体配置保存为 YAML
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper                   # 将 Isaac Lab 环境适配到 RSL-RL 的向量环境接口
    from isaaclab_tasks.utils import get_checkpoint_path                # 根据运行目录和文件名查找待恢复的 checkpoint
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry   # 从任务注册信息读取环境及训练器配置
    import quadruped_robot.tasks                # 导入任务包以执行注册逻辑；无需直接引用导入的模块

    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")    # 读取 --task 对应的环境配置
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")                     # 读取同一任务对应的 RSL-RL 训练配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)         # 用命令行中的种子、运行名和恢复参数覆盖训练配置
    if args_cli.num_envs is not None:                                   # 仅当用户传入 --num_envs 时才覆盖并行环境数量
        env_cfg.scene.num_envs = args_cli.num_envs                      # 设置创建环境时使用的并行实例数

    if args_cli.max_iterations is not None:                             # 仅当用户传入 --max_iterations 时才覆盖默认迭代次数
        agent_cfg.max_iterations = args_cli.max_iterations              # 设置本次训练的 PPO 迭代次数

    env_cfg.seed = agent_cfg.seed                                       # 让仿真环境和训练器使用同一个随机种子

    if args_cli.resume_full_command_range:
        if not agent_cfg.resume:
            raise ValueError("--resume_full_command_range requires --resume.")
        for name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            term_cfg = getattr(env_cfg.curriculum, name, None)
            if term_cfg is None:
                raise ValueError(f"The task has no curriculum term: {name}")
            term_cfg.params["range_multiplier"] = (1.0, 1.0)
        print("[INFO] Resumed training starts with the full velocity-command range.")

    if args_cli.device is not None:                                     # 如果命令行指定了设备，则同步设置仿真和训练设备
        env_cfg.sim.device = args_cli.device                            # 指定物理仿真设备
        agent_cfg.device = args_cli.device                              # 指定 RSL-RL 训练设备

    env_cfg.viewer.origin_type = "asset_root"                           # 使可视化相机的位置跟随机器人根部坐标
    env_cfg.viewer.asset_name = "robot"                                 # 指定相机跟随场景中的 robot 资产
    env_cfg.viewer.env_index = 0                                        # 在第 0 个并行环境中观察机器人
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)                                # 设置相机相对机器人的观察位置
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)                             # 设置相机的观察目标位置

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))    # 日志根目录按实验名称分组；相对路径以启动命令的工作目录为基准
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""))    # 用时间戳和可选运行名区分每次训练
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)         # 创建保存环境和训练器参数快照的目录
    print(f"[INFO] Creating {env_cfg.scene.num_envs} environments for {log_dir}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)                          # 根据环境配置实例化指定任务
    print("[INFO] Environment creation complete; constructing PPO runner.", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # 包装向量环境，并按训练配置选择是否裁剪策略动作
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)     # 创建 PPO 训练器，并将日志写入本次运行目录
    print("[INFO] PPO runner constructed.", flush=True)

    if agent_cfg.resume:                                                # 仅在请求恢复训练时加载已有 checkpoint
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)   # 从指定的历史运行和 checkpoint 参数解析实际文件路径
        print(f"[INFO] Resuming from: {checkpoint}")                    # 在终端显示实际加载的 checkpoint，便于核对
        runner.load(checkpoint, load_optimizer=not args_cli.reset_optimizer)                        # 默认同时恢复优化器；--reset_optimizer 仅恢复策略权重
        if args_cli.reset_optimizer:                                    # 如果优化器已重置，明确打印提示
            print("[INFO] Policy weights restored; optimizer state reset from the current configuration.")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)     # 保存本次实际使用的环境配置，供复现实验和排查问题
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg) # 保存本次实际使用的 RSL-RL 训练配置
    torch.backends.cuda.matmul.allow_tf32 = True                        # 允许 CUDA 矩阵乘法使用 TF32，以提高兼容设备上的计算速度
    print("[INFO] Entering PPO learn loop; the first progress line appears after the first rollout/update.", flush=True)
    if args_cli.debug_startup:
        original_env_step = env.step
        first_step = [True]

        def debug_first_env_step(actions):
            if first_step[0]:
                print("[DIAGNOSTIC] Entering the first vectorized env.step().", flush=True)
            result = original_env_step(actions)
            if first_step[0]:
                print("[DIAGNOSTIC] First vectorized env.step() returned.", flush=True)
                first_step[0] = False
                faulthandler.cancel_dump_traceback_later()
            return result

        env.step = debug_first_env_step
        faulthandler.dump_traceback_later(30, repeat=False)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)      # 启动训练；初始化时随机化各环境的回合进度，错开重置时间
    if args_cli.debug_startup:
        faulthandler.cancel_dump_traceback_later()

    env.close()                                                         # 训练正常结束后关闭环境并释放仿真资源



#==================================================================================
# 直接执行此文件时运行训练；作为模块导入时不启动训练
#==================================================================================
if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
