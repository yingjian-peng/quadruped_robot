"""RSL-RL 训练、播放和评测脚本共用的命令行参数。"""


from __future__ import annotations                      # 延迟解析类型注解，兼容前向引用
import argparse                                         # 定义和组织命令行参数
import random                                           # 在用户传入随机种子 -1 时生成随机种子
import sys                                              # 读取并修改进程命令行参数，供 Isaac Sim Kit 使用

DEFAULT_GPU0_KIT_ARGS = (                               # 集中定义本项目默认使用的 GPU 0 Kit 参数
    "--/renderer/multiGpu/enabled=false "               # 关闭渲染器的多 GPU 模式，固定使用单卡渲染
    "--/renderer/activeGpu=0 "                          # 将渲染任务指定到第 0 张 GPU
    "--/physics/cudaDevice=0"                           # 将物理仿真指定到第 0 张 GPU
)



# ==================================================================================
# 向传入的解析器添加 RSL-RL 项目通用参数
# ==================================================================================
def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl")         # 建立独立参数组，使命令行帮助信息更容易阅读
    group.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint.")  # 指定是否从已有 checkpoint 恢复训练
    group.add_argument("--load_run",type=str,default=None,help="Run directory or regular expression to load.")  # 指定要恢复的运行目录，也支持正则表达式匹配
    group.add_argument("--checkpoint",type=str,default=None,help="Checkpoint file or regular expression to load.")  # 指定要恢复的模型文件，也支持正则表达式匹配
    group.add_argument("--run_name",type=str,default=None,help="Optional suffix for the new log directory.")  # 为新日志目录添加可选后缀，方便区分实验
    group.add_argument("--seed",type=int,default=None,help="Random seed. Use -1 to choose one randomly.")  # 指定随机种子；-1 表示运行时随机生成
    group.add_argument("--reset_optimizer",action="store_true",help="When resuming, load policy weights but reinitialize optimizer state from the current config.")  # 添加恢复训练时重置优化器状态的开关



# ==================================================================================
# 设置默认 GPU 和经过验证的 balanced 渲染预设
# ==================================================================================
def apply_local_app_defaults(args_cli: argparse.Namespace) -> None:
    args_cli.device = (args_cli.device or "cuda:0")                 # 未指定设备时默认使用 CUDA 第 0 张 GPU
    if (getattr(args_cli, "rendering_mode", None) is None):         # 未指定渲染质量时默认使用已验证的 balanced 预设
        args_cli.rendering_mode = "balanced"                        # 将渲染模式设置为 balanced
    kit_args = (getattr(args_cli, "kit_args", "") or "")            # 读取 Kit 参数；属性不存在或值为空时使用空字符串
    default_parts = [part for part in DEFAULT_GPU0_KIT_ARGS.split() if part not in kit_args.split()]    # 找出默认 Kit 参数中尚未由用户提供的部分，避免重复添加
    for part in default_parts:                                      # 逐个检查需要追加到进程命令行的默认参数
        if part not in sys.argv:                                    # 只在参数尚未出现在进程命令行时追加，避免重复设置
            sys.argv.append(part)                                   # 将 GPU 和渲染器默认项传递给 Kit 启动流程



# ==================================================================================
# 将命令行选项覆盖到 RSL-RL 智能体配置中
# ==================================================================================
def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    if args_cli.seed is not None:                   # 用户显式传入种子时，使用该值更新训练配置
        agent_cfg.seed = (random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed)  # 种子 -1 时生成 0 到 10000 之间的随机整数，否则沿用用户指定值
    if args_cli.resume:                             # 只有启用 --resume 时才打开配置中的恢复训练开关
        agent_cfg.resume = True                     # 通知训练脚本从指定 checkpoint 继续
    if args_cli.load_run is not None:               # 用户指定运行目录时，覆盖配置里的待加载运行名
        agent_cfg.load_run = args_cli.load_run      # 保存要查找的历史运行目录或匹配表达式
    if (args_cli.checkpoint is not None):           # 用户指定 checkpoint 时，覆盖配置中的 checkpoint 名称
        agent_cfg.load_checkpoint = (args_cli.checkpoint)  # 保存要加载的模型文件名或匹配表达式
    if args_cli.run_name is not None:               # 用户指定新运行名时，覆盖配置中的默认后缀
        agent_cfg.run_name = (args_cli.run_name)    # 将运行名写入智能体配置，供训练日志目录命名使用
    return agent_cfg                                # 返回完成命令行覆盖后的智能体配置
