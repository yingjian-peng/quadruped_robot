"""Benchmark a locomotion policy against a fixed command matrix.

The script deliberately does not use the keyboard controller.  It evaluates
every command/trial in parallel, disables domain randomization, and writes one
row per rollout plus a category-level summary.  This makes checkpoint-to-
checkpoint comparisons reproducible and exposes weak lateral/yaw behaviours
that are hidden by an aggregate training reward.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from isaaclab.app import AppLauncher


sys.path.append(os.path.dirname(__file__))
import cli_args


PROJECT_ROOT = Path(__file__).resolve().parents[3]


parser = argparse.ArgumentParser(
    description="Benchmark an Isaac Lab RSL-RL locomotion checkpoint."
)
parser.add_argument("--task", default=None, help="Registered Gymnasium task ID.")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Checkpoint file or run directory; defaults to the newest checkpoint under logs/rsl_rl.",
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory for benchmark CSV files."
)
parser.add_argument(
    "--trials", type=int, default=3, help="Independent resets per command (default: 3)."
)
parser.add_argument(
    "--duration_s",
    type=float,
    default=8.0,
    help="Duration of each command rollout in seconds.",
)
parser.add_argument(
    "--warmup_s",
    type=float,
    default=2.0,
    help="Initial time excluded from steady-state metrics.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed used for reproducible simulator resets."
)
parser.add_argument(
    "--fall_height_m",
    type=float,
    default=0.18,
    help="Count a rollout as fallen when base height drops below this value (default: 0.18 m).",
)
parser.add_argument(
    "--fall_tilt_rad",
    type=float,
    default=0.8,
    help="Count a rollout as fallen when body tilt exceeds this value (default: 0.8 rad).",
)
parser.add_argument(
    "--summarize_only",
    action="store_true",
    help="Create summary.csv from an existing rollouts.csv without launching Isaac Sim.",
)
parser.add_argument(
    "--rollouts_path",
    type=str,
    default=None,
    help="Input rollouts.csv for --summarize_only; defaults to the newest benchmark result.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.summarize_only and args_cli.task is None:
    parser.error("--task is required unless --summarize_only is specified")
if args_cli.trials < 1:
    parser.error("--trials must be at least 1")
if args_cli.duration_s <= 0.0 or not 0.0 <= args_cli.warmup_s < args_cli.duration_s:
    parser.error("require duration_s > 0 and 0 <= warmup_s < duration_s")
if args_cli.fall_height_m <= 0.0 or not 0.0 < args_cli.fall_tilt_rad < math.pi:
    parser.error("require fall_height_m > 0 and 0 < fall_tilt_rad < pi")
if not args_cli.summarize_only:
    cli_args.apply_local_app_defaults(args_cli)

simulation_app = None

LITE3_FOOT_NAMES = ("FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT")


@dataclass(frozen=True)
class CommandCase:
    name: str
    category: str
    vx: float
    vy: float
    wz: float


# Values match the manual keyboard test range, with several mixed commands to
# ensure that a policy does not only work on cardinal directions.
COMMAND_CASES = (
    CommandCase("stand", "stand", 0.0, 0.0, 0.0),
    CommandCase("forward", "forward", 0.6, 0.0, 0.0),
    CommandCase("backward", "forward", -0.6, 0.0, 0.0),
    CommandCase("left", "lateral", 0.0, 0.4, 0.0),
    CommandCase("right", "lateral", 0.0, -0.4, 0.0),
    CommandCase("ccw", "yaw", 0.0, 0.0, 1.0),
    CommandCase("cw", "yaw", 0.0, 0.0, -1.0),
    CommandCase("left_ccw", "lateral_yaw", 0.0, 0.3, 0.7),
    CommandCase("right_cw", "lateral_yaw", 0.0, -0.3, -0.7),
    CommandCase("forward_ccw", "forward_yaw", 0.5, 0.0, 0.7),
    CommandCase("forward_cw", "forward_yaw", 0.5, 0.0, -0.7),
)


def disable_randomization(env_cfg) -> None:
    """Use one nominal robot/ground setup for an apples-to-apples benchmark."""
    for name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_actuator_gains",
        "randomize_push_robot",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    if hasattr(env_cfg.events, "randomize_reset_base"):
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
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

    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1_000_000.0, 1_000_000.0)
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "terrain_out_of_bounds"):
        env_cfg.terminations.terrain_out_of_bounds = None


def resolve_input_path(value: str) -> Path:
    """Resolve an input from either the caller's cwd or the rl_training root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()


def resolve_output_path(value: str) -> Path:
    """Keep relative benchmark output paths anchored at the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(agent_cfg) -> str:
    if args_cli.checkpoint_path is not None:
        checkpoint_path = resolve_input_path(args_cli.checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoints = sorted(
                checkpoint_path.glob("model_*.pt"),
                key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime),
            )
            if not checkpoints:
                raise FileNotFoundError(
                    f"No model_*.pt checkpoint found in: {checkpoint_path}"
                )
            return str(checkpoints[-1])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return str(checkpoint_path)

    log_root = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    checkpoints = sorted(
        log_root.rglob("model_*.pt"),
        key=lambda path: (path.stat().st_mtime, checkpoint_iteration(path)),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under: {log_root}")
    return str(checkpoints[-1])


def resolve_rollouts_path() -> Path:
    """Resolve an explicit rollouts file or select the newest benchmark run."""
    if args_cli.rollouts_path is not None:
        path = resolve_input_path(args_cli.rollouts_path)
        if not path.is_file():
            raise FileNotFoundError(f"Rollout metrics do not exist: {path}")
        return path
    if args_cli.output_dir is not None:
        path = resolve_output_path(args_cli.output_dir) / "rollouts.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Rollout metrics do not exist: {path}")
        return path
    candidates = list(
        (PROJECT_ROOT / "logs" / "rsl_rl").glob("*/*/benchmark/rollouts.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            "No benchmark/rollouts.csv found under logs/rsl_rl; pass --rollouts_path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def unwrap_observation(observation):
    """RSL-RL wrappers may return either obs or (obs, extras)."""
    return observation[0] if isinstance(observation, tuple) else observation


def build_foot_indices(robot, contact_sensor):
    """Resolve Lite3 foot indices in the phase-clock order."""
    asset_ids, asset_names = robot.find_bodies(
        list(LITE3_FOOT_NAMES), preserve_order=True
    )
    sensor_ids, sensor_names = contact_sensor.find_bodies(
        list(LITE3_FOOT_NAMES), preserve_order=True
    )
    if (
        tuple(asset_names) != LITE3_FOOT_NAMES
        or tuple(sensor_names) != LITE3_FOOT_NAMES
    ):
        raise RuntimeError(
            "Could not resolve all Lite3 feet in FL/FR/HL/HR order: "
            f"asset={asset_names}, sensor={sensor_names}"
        )
    return list(LITE3_FOOT_NAMES), asset_ids, sensor_ids


def evaluate_batch(env, policy, runs, foot_names, asset_foot_ids, sensor_foot_ids):
    """Evaluate all command/trial pairs in parallel during one simulator run.

    Running every case serially can exceed the short lifetime of some headless
    Kit configurations.  A vectorized rollout is also faster and gives every
    command the same reset state and simulation conditions.
    """
    import torch

    obs = unwrap_observation(env.reset())
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    command_term = unwrapped.command_manager.get_term("base_velocity")
    commands = torch.tensor(
        [[case.vx, case.vy, case.wz] for _, case in runs],
        device=unwrapped.device,
        dtype=torch.float32,
    )
    command_term.vel_command_b[:] = commands

    total_steps = round(args_cli.duration_s / unwrapped.step_dt)
    warmup_steps = round(args_cli.warmup_s / unwrapped.step_dt)
    num_envs = len(runs)
    measurement_start_position = torch.zeros(num_envs, 2, device=unwrapped.device)
    measurement_last_position = torch.zeros_like(measurement_start_position)
    measurement_started = torch.zeros(
        num_envs, dtype=torch.bool, device=unwrapped.device
    )
    sample_count = torch.zeros(num_envs, device=unwrapped.device)
    gait_sample_count = torch.zeros(num_envs, device=unwrapped.device)
    moving_command = torch.linalg.vector_norm(commands, dim=1) > 1.0e-4
    previous_action = None
    previous_lin_vel_z = None
    gait_metric_names = {
        "gait_clock_match",
        "diag_sync_fl_hr",
        "diag_sync_fr_hl",
        "diag_antiphase",
    }
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
            "base_height_m",
            "vertical_acc_abs",
            "action_rate_l2",
            "joint_deviation_mean_abs_rad",
            "joint_power_w",
            "gait_clock_match",
            "diag_sync_fl_hr",
            "diag_sync_fr_hl",
            "diag_antiphase",
        )
    }
    stance_slip_sum = torch.zeros(num_envs, device=unwrapped.device)
    stance_contact_count = torch.zeros(num_envs, device=unwrapped.device)
    swing_height_sum = torch.zeros(num_envs, device=unwrapped.device)
    swing_contact_sum = torch.zeros(num_envs, device=unwrapped.device)
    swing_sample_count = torch.zeros(num_envs, device=unwrapped.device)
    duty_sum = torch.zeros(num_envs, len(foot_names), device=unwrapped.device)
    terminated = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    fall_detected = torch.zeros_like(terminated)

    for step in range(total_steps):
        command_term.vel_command_b[:] = commands
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        command_term.vel_command_b[:] = commands
        obs = unwrap_observation(obs)
        dones = torch.as_tensor(dones, device=unwrapped.device, dtype=torch.bool)
        gravity_b = robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
        fell = (robot.data.root_pos_w[:, 2] < args_cli.fall_height_m) | (
            tilt > args_cli.fall_tilt_rad
        )
        fall_detected |= fell & ~terminated & ~dones
        # ManagerBasedRLEnv auto-resets completed environments in env.step().
        # Exclude reset states, falls, and every later state from this rollout.
        valid = ~(terminated | dones | fell)
        terminated |= dones | fell

        # Metrics begin only after the gait has had time to settle.
        if step < warmup_steps:
            previous_action = unwrapped.action_manager.action.clone()
            previous_lin_vel_z = robot.data.root_lin_vel_b[:, 2].clone()
            continue

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

        actual = robot.data.root_lin_vel_b
        angular = robot.data.root_ang_vel_b
        command_error = commands - torch.stack(
            (actual[:, 0], actual[:, 1], angular[:, 2]), dim=1
        )
        samples["mae_vx"] += command_error[:, 0].abs() * valid_float
        samples["mae_vy"] += command_error[:, 1].abs() * valid_float
        samples["mae_wz"] += command_error[:, 2].abs() * valid_float
        samples["actual_vx"] += actual[:, 0] * valid_float
        samples["actual_vy"] += actual[:, 1] * valid_float
        samples["actual_wz"] += angular[:, 2] * valid_float
        samples["vertical_speed_abs"] += actual[:, 2].abs() * valid_float
        samples["base_height_m"] += robot.data.root_pos_w[:, 2] * valid_float

        samples["tilt_rad"] += tilt * valid_float
        if previous_lin_vel_z is not None:
            vertical_acc = (actual[:, 2] - previous_lin_vel_z) / unwrapped.step_dt
            samples["vertical_acc_abs"] += vertical_acc.abs() * valid_float
        previous_lin_vel_z = actual[:, 2].clone()

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

        # Flat-ground swing diagnostics.  A swing foot that is still in
        # contact contributes to swing_scuff_rate; its world z position is
        # averaged separately so clock transition steps do not bias the mean.
        expected_swing = ~expected_stance
        valid_swing = expected_swing & gait_valid.unsqueeze(1)
        swing_height_sum += (
            robot.data.body_pos_w[:, asset_foot_ids, 2] * valid_swing
        ).sum(dim=1)
        swing_contact_sum += (contact & valid_swing).sum(dim=1)
        swing_sample_count += valid_swing.sum(dim=1)

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
        samples["joint_power_w"] += (
            torch.abs(robot.data.joint_vel * robot.data.applied_torque).sum(dim=1)
            * valid_float
        )

    def divide_or_nan(numerator, denominator):
        denominator = denominator.expand_as(numerator)
        return torch.where(
            denominator > 0,
            numerator / torch.clamp(denominator, min=1.0),
            torch.full_like(numerator, float("nan")),
        )

    samples = {
        name: divide_or_nan(
            values, gait_sample_count if name in gait_metric_names else sample_count
        )
        for name, values in samples.items()
    }
    stance_slip = divide_or_nan(stance_slip_sum, stance_contact_count)
    swing_height = divide_or_nan(swing_height_sum, swing_sample_count)
    swing_scuff_rate = divide_or_nan(swing_contact_sum, swing_sample_count)
    duty = divide_or_nan(duty_sum, sample_count.unsqueeze(1))
    measured_time = sample_count * unwrapped.step_dt
    displacement = divide_or_nan(
        torch.linalg.vector_norm(
            measurement_last_position - measurement_start_position, dim=1
        ),
        measured_time,
    )

    rows = []
    for index, (trial, case) in enumerate(runs):
        row = {
            "case": case.name,
            "category": case.category,
            "cmd_vx": case.vx,
            "cmd_vy": case.vy,
            "cmd_wz": case.wz,
            "trial": trial,
            "terminated": int(terminated[index].item()),
            "fall_detected": int(fall_detected[index].item()),
            "world_displacement_mps": float(displacement[index]),
            "yaw_translation_drift_mps": float(displacement[index])
            if case.category == "yaw"
            else float("nan"),
            "measured_duration_s": float(measured_time[index]),
            "stance_slip_mps": float(stance_slip[index]),
            "swing_height_m": float(swing_height[index]),
            "swing_scuff_rate": float(swing_scuff_rate[index]),
        }
        row.update({name: float(values[index]) for name, values in samples.items()})
        row.update(
            {
                f"duty_{name}": float(duty[index, foot_index])
                for foot_index, name in enumerate(foot_names)
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    summary = []
    for category, group in sorted(grouped.items()):
        result: dict[str, object] = {"category": category, "rollouts": len(group)}
        numeric_fields = [key for key in group[0] if key not in {"case", "category"}]
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in group
                if isinstance(row.get(field), (int, float))
                and math.isfinite(float(row[field]))
            ]
            if values:
                result[f"{field}_mean"] = mean(values)
                result[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def summarize_rollouts_file(rollouts_path: Path) -> Path:
    """Generate the category summary without starting the Isaac Sim application."""
    with rollouts_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for source_row in reader:
            case = source_row.pop("case", None)
            category = source_row.pop("category", None)
            if case is None or category is None:
                raise ValueError(f"Missing case/category columns in: {rollouts_path}")
            row = {"case": case, "category": category}
            row.update(
                {
                    name: float(value) if value not in (None, "") else float("nan")
                    for name, value in source_row.items()
                }
            )
            rows.append(row)
    if not rows:
        raise ValueError(f"No rollout rows found in: {rollouts_path}")
    summary_path = rollouts_path.with_name("summary.csv")
    write_csv(summary_path, summarize(rows))
    return summary_path


def main() -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import quadruped_robot.tasks  # noqa: F401

    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    runs = [(trial, case) for trial in range(args_cli.trials) for case in COMMAND_CASES]
    env_cfg.scene.num_envs = len(runs)
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    disable_randomization(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    checkpoint = resolve_checkpoint(agent_cfg)
    output_dir = (
        resolve_output_path(args_cli.output_dir)
        if args_cli.output_dir is not None
        else Path(checkpoint).parent / "benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "rollouts.csv"
    summary_path = output_dir / "summary.csv"
    print(f"[BENCHMARK] loading checkpoint: {checkpoint}")
    env = RslRlVecEnvWrapper(
        gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions
    )
    try:
        print("[BENCHMARK] creating RSL-RL runner")
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        print("[BENCHMARK] restoring policy weights")
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print("[BENCHMARK] resolving foot contacts")
        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        foot_names, asset_foot_ids, sensor_foot_ids = build_foot_indices(
            robot, contact_sensor
        )

        print(f"[BENCHMARK] evaluating {len(runs)} rollouts in parallel", flush=True)
        rows = evaluate_batch(
            env, policy, runs, foot_names, asset_foot_ids, sensor_foot_ids
        )
        write_csv(raw_path, rows)
        write_csv(summary_path, summarize(rows))
        for row in rows:
            print(
                f"[BENCHMARK] trial={row['trial']} case={row['case']} "
                f"mae(vx,vy,wz)=({row['mae_vx']:.3f}, {row['mae_vy']:.3f}, {row['mae_wz']:.3f})",
                flush=True,
            )

        print(f"[BENCHMARK] checkpoint: {checkpoint}")
        print(f"[BENCHMARK] rollout metrics: {raw_path}")
        print(f"[BENCHMARK] category summary: {summary_path}")
    finally:
        env.close()


if __name__ == "__main__":
    if args_cli.summarize_only:
        rollouts_path = resolve_rollouts_path()
        print(f"[BENCHMARK] rollout metrics: {rollouts_path}")
        summary_path = summarize_rollouts_file(rollouts_path)
        print(f"[BENCHMARK] category summary: {summary_path}")
    else:
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        try:
            main()
        finally:
            simulation_app.close()
