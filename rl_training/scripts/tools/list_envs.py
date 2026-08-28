"""List the Isaac Lab task IDs provided by this repository."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym

    import robot_lab.tasks  # noqa: F401

    for task_id in sorted(task_id for task_id in gym.registry if task_id_prefix(task_id)):
        print(task_id)


def task_id_prefix(task_id: str) -> bool:
    return task_id.startswith("RobotLab-Isaac-Velocity-")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
