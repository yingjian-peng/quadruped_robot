"""Convert a project-owned wheel-legged quadruped URDF to its task USD.

Run through ``isaaclab.sh -p``.  This tool never overwrites an existing USD
without ``--force`` so an inspected simulation asset is not replaced by accident.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_MODELS_DIR = PROJECT_ROOT / "robot_models"
ASSETS = {
    "deeprobotics_m20": (
        ROBOT_MODELS_DIR / "deeprobotics/m20_description/urdf/m20.urdf",
        ROBOT_MODELS_DIR / "deeprobotics/m20_description/usd/m20.usd",
    ),
    "unitree_b2w": (
        ROBOT_MODELS_DIR / "unitree/b2w_description/urdf/b2w_description.urdf",
        ROBOT_MODELS_DIR / "unitree/b2w_description/usd/b2w_description.usd",
    ),
    "unitree_go2w": (
        ROBOT_MODELS_DIR / "unitree/go2w_description/urdf/go2w_description.urdf",
        ROBOT_MODELS_DIR / "unitree/go2w_description/usd/go2w_description.usd",
    ),
}

parser = argparse.ArgumentParser(description="Convert a wheel-quadruped URDF to its project-owned USD path.")
parser.add_argument("robot", choices=sorted(ASSETS), help="Robot asset to convert.")
parser.add_argument("--force", action="store_true", help="Overwrite an existing destination USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    source_path, destination_path = ASSETS[args_cli.robot]
    if not source_path.is_file():
        raise FileNotFoundError(f"URDF source does not exist: {source_path}")
    if destination_path.exists() and not args_cli.force:
        raise FileExistsError(f"USD already exists: {destination_path}. Re-run with --force to replace it.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    converter_cfg = UrdfConverterCfg(
        asset_path=str(source_path),
        usd_dir=str(destination_path.parent),
        usd_file_name=destination_path.name,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=True,
        replace_cylinders_with_capsules=False,
        force_usd_conversion=args_cli.force,
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="none"),
    )
    converter = UrdfConverter(converter_cfg)
    if Path(converter.usd_path) != destination_path or not destination_path.is_file():
        raise RuntimeError(f"USD conversion did not produce the expected path: {destination_path}")
    print(f"USD_CONVERSION_OK robot={args_cli.robot} source={source_path} output={destination_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
