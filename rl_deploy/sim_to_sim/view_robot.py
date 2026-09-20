#!/usr/bin/env python3
"""Interactively inspect the project-owned Lite3 MuJoCo model.

Default mode is a kinematic, zero-gravity pose viewer.  ``--gravity`` steps
the model so Lite3 settles on the display floor.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - environment hint for the user
    sys.exit(
        "MuJoCo is not installed in this interpreter. Run with:\n"
        "  /home/robot/anaconda3/envs/pyj_rl_simtosim/bin/python " + " ".join(sys.argv)
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MJCF_DIR = PROJECT_ROOT / "robot_models/deeprobotics/Lite3/Lite3_mjcf/mjcf"
LITE3_MJCF = MJCF_DIR / "Lite3.xml"
ENVIRONMENT_MJCF = MJCF_DIR / "view_environment.xml"
CACHE_DIR = Path(os.environ.get("QUADRUPED_MUJOCO_CACHE", Path.home() / ".cache/quadruped_robot/mujoco"))
BASE_HEIGHT = 0.50
_MESH_FILE = re.compile(r'(<mesh\b[^>]*?\bfile\s*=\s*")([^"]+)(")')


def build_display_mjcf() -> Path:
    """Build a cache-only entry MJCF which includes the display environment."""
    if not LITE3_MJCF.is_file() or not ENVIRONMENT_MJCF.is_file():
        raise FileNotFoundError(f"missing Lite3 model or display environment under {MJCF_DIR}")
    text = LITE3_MJCF.read_text(encoding="utf-8")

    def absolute_mesh(match: re.Match) -> str:
        path = (LITE3_MJCF.parent / match.group(2)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Lite3 mesh not found: {path}")
        return match.group(1) + str(path) + match.group(3)

    text = _MESH_FILE.sub(absolute_mesh, text)
    closing_tag = "</mujoco>"
    if closing_tag not in text:
        raise ValueError(f"invalid MJCF, missing {closing_tag}: {LITE3_MJCF}")
    include = f'  <include file="{ENVIRONMENT_MJCF}"/>\n'
    text = text.rsplit(closing_tag, 1)[0] + include + closing_tag

    target = CACHE_DIR / "Lite3.display.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def load_model() -> tuple["mujoco.MjModel", Path]:
    display_mjcf = build_display_mjcf()
    model = mujoco.MjModel.from_xml_path(str(display_mjcf))
    return model, display_mjcf


def set_base_height(model: "mujoco.MjModel", data: "mujoco.MjData") -> None:
    free_joint = next(
        (index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE),
        None,
    )
    if free_joint is None:
        raise ValueError("Lite3 MJCF has no floating-base joint")
    data.qpos[model.jnt_qposadr[free_joint] + 2] = BASE_HEIGHT


def camera_for_robot(model: "mujoco.MjModel", data: "mujoco.MjData") -> "mujoco.MjvCamera":
    free_joint = next(
        (index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE),
        None,
    )
    if free_joint is None:
        center, radius = np.zeros(3), 1.0
    else:
        inside = np.zeros(model.nbody, dtype=bool)
        inside[model.jnt_bodyid[free_joint]] = True
        for body in range(int(model.jnt_bodyid[free_joint]) + 1, model.nbody):
            inside[model.body_parentid[body]] |= inside[body]
        geoms = [index for index in range(model.ngeom) if inside[model.geom_bodyid[index]]]
        points = np.asarray([data.geom_xpos[index] for index in geoms])
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        radius = max(float(np.max(np.linalg.norm(points - center, axis=1))), 0.1)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = center
    camera.distance = 2.8 * radius
    camera.azimuth, camera.elevation = 135.0, -18.0
    return camera


def describe(model: "mujoco.MjModel") -> str:
    kinds = Counter(mujoco.mjtJoint(model.jnt_type[i]).name.replace("mjJNT_", "").lower() for i in range(model.njnt))
    return f"nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom} joints[{dict(kinds)}]"


def run_viewer(model: "mujoco.MjModel", data: "mujoco.MjData", gravity: bool) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        camera = camera_for_robot(model, data)
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            if gravity:
                started = time.time()
                mujoco.mj_step(model, data)
            else:
                # The right-hand Joint sliders edit qpos.  This only refreshes
                # kinematics and does not write motor controls or integrate.
                mujoco.mj_forward(model, data)
            viewer.sync()
            if gravity:
                time.sleep(max(0.0, model.opt.timestep - (time.time() - started)))
            else:
                time.sleep(0.01)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View Lite3 in MuJoCo.")
    parser.add_argument("--gravity", action="store_true", help="Step the model with gravity enabled.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model, display_mjcf = load_model()
        data = mujoco.MjData(model)
        if not args.gravity:
            model.opt.gravity[:] = 0.0
        set_base_height(model, data)
        mujoco.mj_forward(model, data)
    except (FileNotFoundError, ValueError, mujoco.FatalError) as error:
        print(f"[view_robot] {error}", file=sys.stderr)
        return 1

    print(f"[view_robot] model={LITE3_MJCF}")
    print(f"[view_robot] environment={ENVIRONMENT_MJCF}")
    print(f"[view_robot] display entry={display_mjcf}")
    print(f"[view_robot] base_z={BASE_HEIGHT:.2f} gravity={'on' if args.gravity else 'off'} {describe(model)}")
    try:
        run_viewer(model, data, args.gravity)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
