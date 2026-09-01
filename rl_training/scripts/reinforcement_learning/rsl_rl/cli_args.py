"""Shared command-line options for the local RSL-RL entry points."""

from __future__ import annotations

import argparse
import random
import sys


DEFAULT_GPU0_KIT_ARGS = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/activeGpu=0 "
    "--/physics/cudaDevice=0"
)


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint.")
    group.add_argument("--load_run", type=str, default=None, help="Run directory or regular expression to load.")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file or regular expression to load.")
    group.add_argument("--run_name", type=str, default=None, help="Optional suffix for the new log directory.")
    group.add_argument("--seed", type=int, default=None, help="Random seed. Use -1 to choose one randomly.")


def apply_local_app_defaults(args_cli: argparse.Namespace) -> None:
    """Use GPU0 by default without changing Isaac Sim's render quality preset."""
    args_cli.device = args_cli.device or "cuda:0"
    kit_args = getattr(args_cli, "kit_args", "") or ""
    default_parts = [part for part in DEFAULT_GPU0_KIT_ARGS.split() if part not in kit_args.split()]
    for part in default_parts:
        if part not in sys.argv:
            sys.argv.append(part)


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    if args_cli.seed is not None:
        agent_cfg.seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    return agent_cfg
