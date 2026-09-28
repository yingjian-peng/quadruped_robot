"""Compare Isaac Lab locomotion benchmark summaries across training runs.

``benchmark.py`` writes ``benchmark/summary.csv`` next to every checkpoint it evaluates.
This tool lines those summaries up so a Lite3 configuration change can be judged against
the previous run without re-launching Isaac Sim.

Examples
--------
Compare two explicit runs::

    python3 scripts/tools/compare_benchmarks.py \
        logs/rsl_rl/deeprobotics_lite3_flat/<baseline-run> \
        logs/rsl_rl/deeprobotics_lite3_flat/<candidate-run>

Compare every benchmarked run under the Lite3 flat log root and export a CSV::

    python3 scripts/tools/compare_benchmarks.py --root logs/rsl_rl/deeprobotics_lite3_flat \
        --csv /tmp/lite3_bench_table.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Metrics printed by default: (column prefix in summary.csv, display label).
METRICS: tuple[tuple[str, str], ...] = (
    ("mae_vx", "|e_vx| [m/s]"),
    ("mae_vy", "|e_vy| [m/s]"),
    ("mae_wz", "|e_wz| [rad/s]"),
    ("stance_slip_mps", "foot slip [m/s]"),
    ("tilt_rad", "body tilt [rad]"),
    ("action_rate_l2", "action rate l2"),
    ("joint_power_w", "joint power [W]"),
    ("gait_clock_match", "gait clock match"),
    ("vertical_acc_abs", "vertical acc"),
    ("world_displacement_mps", "displacement [m/s]"),
)

CATEGORIES = ("stand", "forward", "forward_yaw", "lateral", "lateral_yaw", "yaw")
LOCOMOTION = tuple(category for category in CATEGORIES if category != "stand")

class RunSummary:
    """Category-level rows of one ``summary.csv`` file."""

    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        self.rows: dict[str, dict[str, float]] = {}
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                self.rows[row["category"]] = row

    def value(self, category: str, metric: str) -> float:
        row = self.rows.get(category)
        if row is None:
            return float("nan")
        try:
            return float(row[f"{metric}_mean"])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def average(self, categories: tuple[str, ...], metric: str) -> float:
        values = [self.value(category, metric) for category in categories]
        values = [value for value in values if value == value]  # drop NaN
        return sum(values) / len(values) if values else float("nan")


def resolve_summary(path: Path) -> tuple[str, Path] | None:
    """Accept a run directory, a benchmark directory or a summary.csv path."""
    candidates = (path / "summary.csv", path / "benchmark" / "summary.csv", path)
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "summary.csv":
            label = path.parent.name if path.name == "summary.csv" else path.name
            if label == "benchmark":
                label = path.parent.name
            return label, candidate
    return None


def discover(root: Path) -> list[tuple[str, Path]]:
    """All benchmarked runs under ``root``, oldest first."""
    found = [(summary.parent.parent.name, summary) for summary in root.glob("*/benchmark/summary.csv")]
    return sorted(found, key=lambda item: item[1].stat().st_mtime)


def format_value(value: float, delta: float) -> str:
    if value != value:
        return "n/a"
    if delta != delta or delta == 0.0:
        return f"{value:.3f}"
    return f"{value:.3f} ({delta:+.0%})"


def print_metric_table(metric: str, label: str, runs: list[RunSummary], baseline: int) -> None:
    print(f"\n== {metric} — {label} ==")
    width = max(len("locomotion avg"), max(len(category) for category in CATEGORIES))
    header = " " * (width + 2) + "".join(f"{run.label[:26]:>28}" for run in runs)
    print(header)
    for category in list(CATEGORIES) + ["locomotion avg"]:
        values = [
            run.average(LOCOMOTION, metric) if category == "locomotion avg" else run.value(category, metric)
            for run in runs
        ]
        cells = ""
        for index, value in enumerate(values):
            delta = float("nan")
            if index != baseline and values[baseline] == values[baseline] and values[baseline] != 0.0:
                delta = (value - values[baseline]) / abs(values[baseline])
            cells += f"{format_value(value, delta):>28}"
        print(f"{category:<{width}}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="Run / benchmark directories or summary.csv files.")
    parser.add_argument("--root", type=Path, default=None, help="Discover all benchmarked runs under this log root.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Subset of metric keys to print.")
    parser.add_argument("--baseline", type=int, default=0, help="Index of the run used to compute deltas.")
    parser.add_argument("--csv", type=Path, default=None, help="Write a long-form comparison table to this CSV.")
    args = parser.parse_args()

    discovered = discover(args.root) if args.root else []
    requested = [item for path in args.paths if (item := resolve_summary(path)) is not None]
    resolved = discovered + [item for item in requested if item not in discovered]
    if not resolved:
        parser.error("No benchmark summary found; pass run directories or --root.")

    runs = [RunSummary(label, path) for label, path in resolved]
    metrics = [(key, label) for key, label in METRICS if args.metrics is None or key in args.metrics]

    print("Benchmark comparison")
    for run in runs:
        print(f"  {run.label:<40} {run.path}")
    for metric, label in metrics:
        print_metric_table(metric, label, runs, args.baseline)
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["run", "category", "metric", "value"])
            for run in runs:
                for category in list(CATEGORIES) + ["locomotion avg"]:
                    for metric, _ in METRICS:
                        value = (
                            run.average(LOCOMOTION, metric)
                            if category == "locomotion avg"
                            else run.value(category, metric)
                        )
                        if value == value:
                            writer.writerow([run.label, category, metric, f"{value:.6f}"])
        print(f"\n[compare_benchmarks] wrote {args.csv}")


if __name__ == "__main__":
    main()
