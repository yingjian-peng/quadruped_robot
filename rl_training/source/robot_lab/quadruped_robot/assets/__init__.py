# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

from pathlib import Path

import toml

##
# Configuration for different assets.
##

_ASSETS_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "robot_models").is_dir():
            return path
    raise FileNotFoundError("Could not locate the shared robot_models directory.")


ISAACLAB_ASSETS_EXT_DIR = str(_ASSETS_PACKAGE_DIR.parents[1])
"""Path to the extension source directory."""

ROBOT_MODELS_DIR = str(_find_repo_root(_ASSETS_PACKAGE_DIR) / "robot_models")
"""Path to the shared robot model directory."""

ISAACLAB_ASSETS_METADATA = toml.load(Path(ISAACLAB_ASSETS_EXT_DIR) / "config" / "extension.toml")
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]
