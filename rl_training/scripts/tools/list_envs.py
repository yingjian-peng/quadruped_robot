"""List the Isaac Lab task IDs provided by this repository."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym

    import loco_manipulation_lab.tasks  # noqa: F401

    for task_id in sorted(spec.id for spec in gym.registry.values() if task_id_prefix(spec.id)):
        print(task_id)


def task_id_prefix(task_id: str) -> bool:
    project_task_prefixes = (
        "Go2-Arx-LocoManip-",
        "Go2-RearLeg-Balance-",
        "B2W-Z1-LocoManip-",
        "B2W-WheelQuadruped-",
        "Go2W-WheelQuadruped-",
    )
    return task_id.startswith(project_task_prefixes)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
