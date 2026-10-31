"""Print all Unitree Go2 environments registered by this package (and unitree_rl_lab).

Use this to verify the installation:

    isaaclab.sh -p src/utils/list_envs.py --viz none

The 6 tasks of this package (Unitree-Go2-Velocity-Stairs / -StairsV2 /
-Obstacle / -ObstacleHM / -ObstacleHMDiverse / -ObstacleExt)
must appear alongside the upstream unitree_rl_lab tasks.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from prettytable import PrettyTable

import unitree_rl_lab.tasks  # noqa: F401
import tools  # noqa: F401


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Unitree Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
            index += 1
            cfg = task_spec.kwargs.get("env_cfg_entry_point", "")
            table.add_row([index, task_spec.id, task_spec.entry_point, cfg])
    print(table)


if __name__ == "__main__":
    main()
    simulation_app.close()
