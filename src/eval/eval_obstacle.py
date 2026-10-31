"""Evaluate the Unitree-Go2-Velocity-Obstacle policy on obstacles-only terrain.

Spawns envs uniformly across 10 obstacle-density rows (4 -> 12 pillars per
8x8 m tile, 0.35 -> 0.6 m wide, cfg height 1.5 m = 0.75 m above ground since the
box is centred at z=0 — the training curriculum) and
commands a fixed straight-ahead velocity, so the policy must leave the
commanded path on its own to get through. Reports per-difficulty collision
rate (base-contact terminations), other falls, velocity, and lidar clearance.

Usage:
    isaaclab.sh -p src/eval/eval_obstacle.py --viz none --num_envs 200 --steps 2000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--steps", type=int, default=2000, help="Control steps to simulate (50 Hz).")
parser.add_argument("--vx", type=float, default=0.5, help="Commanded forward velocity [m/s].")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model .pt (default: latest run).")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity-Obstacle")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from importlib import metadata

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
import tools  # noqa: F401
from tools.agents.rsl_rl_ppo_cfg import AsymmetricPPORunnerCfg
from tools.stairs.velocity_stairs_env_cfg import RobotStairsV2EnvCfg
from tools.stairs_avoid import extgrid_mdp, heightmap_mdp, obstacle_mdp
from tools.stairs_avoid.velocity_heightmap_env_cfg import (
    RobotHeightmapDiverseEnvCfg,
    RobotHeightmapEnvCfg,
)
from tools.stairs_avoid.velocity_extgrid_env_cfg import (
    RobotObstacleExtEnvCfg,
)
from tools.stairs_avoid.velocity_obstacle_env_cfg import (
    LIDAR_MAX_RANGE,
    RobotObstacleEnvCfg,
)

TASK_CFGS = {
    "Unitree-Go2-Velocity-Obstacle": RobotObstacleEnvCfg,
    # StairsV2 has no lidar: it runs the same pillar course blind (height_scan only)
    "Unitree-Go2-Velocity-StairsV2": RobotStairsV2EnvCfg,
    # height-map variant: clearance column = distance to nearest obstacle cell
    "Unitree-Go2-Velocity-ObstacleHM": RobotHeightmapEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHMDiverse": RobotHeightmapDiverseEnvCfg,
    # extended fine grid: single height_scanner (609 rays), clearance from ext obstacle cells
    "Unitree-Go2-Velocity-ObstacleExt": RobotObstacleExtEnvCfg,
}
NUM_LEVELS = 10
OBJ_MIN, OBJ_MAX = 4, 12  # pillars per tile at difficulty 0 / 1 (training values)


def main():
    env_cfg = TASK_CFGS[args_cli.task]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.robot.actuators["GO2HV"].armature = 0.0

    # obstacles-only terrain: 10 rows ordered sparse -> dense (curriculum grid)
    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=NUM_LEVELS,
        num_cols=4,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "obstacles": terrain_gen.MeshRepeatedBoxesTerrainCfg(
                proportion=1.0,
                object_type="box",
                object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=OBJ_MIN, height=1.5, size=(0.35, 0.35)
                ),
                object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=OBJ_MAX, height=1.5, size=(0.6, 0.6)
                ),
                abs_height_noise=(0.0, 0.0),
                platform_width=1.5,
                platform_height=0.0,
            ),
        },
    )
    env_cfg.scene.terrain.max_init_terrain_level = None  # spawn across all rows
    env_cfg.curriculum.terrain_levels = None  # freeze level assignment
    env_cfg.curriculum.lin_vel_cmd_levels = None  # freeze command range

    # fixed straight-ahead command: the policy must deviate on its own
    cmd = env_cfg.commands.base_velocity
    cmd.rel_standing_envs = 0.0
    cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)

    agent_cfg = AsymmetricPPORunnerCfg()
    agent_cfg.experiment_name = args_cli.task.lower().replace("-", "_")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[EVAL] Loading checkpoint: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    levels = env.unwrapped.scene.terrain.terrain_levels.clone()  # per-env row index
    n = env.unwrapped.num_envs
    term_manager = env.unwrapped.termination_manager

    collisions = torch.zeros(n, device=device)  # base-contact terminations
    other_falls = torch.zeros(n, device=device)  # bad-orientation terminations
    timeouts = torch.zeros(n, device=device)
    vx_sum = torch.zeros(n, device=device)
    clearance_sum = torch.zeros(n, device=device)
    step_cnt = 0

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            hit = term_manager.get_term("base_contact").to(device).bool()
            tipped = term_manager.get_term("bad_orientation").to(device).bool()
            to = extras["time_outs"].to(device).bool() if "time_outs" in extras else torch.zeros_like(dones).bool()
            collisions += hit.float()
            other_falls += (tipped & ~hit).float()
            timeouts += (dones.to(device).bool() & to & ~hit & ~tipped).float()
            vx_sum += env.unwrapped.scene["robot"].data.root_lin_vel_b[:, 0]
            sensors = env.unwrapped.scene.sensors
            if "lidar_scanner" in sensors:
                _, dist = obstacle_mdp._lidar_distances(sensors["lidar_scanner"], LIDAR_MAX_RANGE)
                clearance_sum += dist.min(dim=1).values
            elif args_cli.task == "Unitree-Go2-Velocity-ObstacleExt":
                mask = extgrid_mdp.ext_obstacle_cells(env.unwrapped, SceneEntityCfg("height_scanner"))
                cell_dist, _ = extgrid_mdp._ext_dist_yaw(mask.device)
                d = torch.where(mask, cell_dist.unsqueeze(0), torch.full_like(mask, extgrid_mdp.EXT_MAX_RANGE, dtype=cell_dist.dtype))
                clearance_sum += d.flatten(1).min(dim=1).values
            elif "coarse_scanner" in sensors:
                mask = heightmap_mdp.obstacle_cells(
                    env.unwrapped, SceneEntityCfg("coarse_scanner"), SceneEntityCfg("height_scanner")
                )
                cell_dist, _ = heightmap_mdp._cell_dist_yaw(mask.device)
                d = torch.where(mask, cell_dist.unsqueeze(0), torch.full_like(mask, heightmap_mdp.COARSE_MAX_RANGE, dtype=cell_dist.dtype))
                clearance_sum += d.flatten(1).min(dim=1).values
            step_cnt += 1

    print("\n[EVAL] ==== obstacles-only evaluation ====")
    print(f"[EVAL] command vx={args_cli.vx} m/s (straight), horizon={args_cli.steps} steps ({args_cli.steps / 50:.0f}s)")
    header = f"{'level':>5} {'pillars':>7} {'#envs':>6} {'episodes':>9} {'collisions':>10} {'coll_rate':>9} {'falls':>6} {'mean_vx':>8} {'clearance[m]':>12}"
    print(f"[EVAL] {header}")
    for lv in range(NUM_LEVELS):
        mask = levels == lv
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        diff = lv / max(NUM_LEVELS - 1, 1)
        pillars = OBJ_MIN + diff * (OBJ_MAX - OBJ_MIN)
        c = float(collisions[mask].sum())
        f = float(other_falls[mask].sum())
        t = float(timeouts[mask].sum())
        eps = c + f + t
        rate = c / eps if eps > 0 else float("nan")
        mvx = float((vx_sum[mask] / step_cnt).mean())
        sensors = env.unwrapped.scene.sensors
        has_range = (
            "lidar_scanner" in sensors or "coarse_scanner" in sensors
            or args_cli.task == "Unitree-Go2-Velocity-ObstacleExt"
        )
        mcl = f"{float((clearance_sum[mask] / step_cnt).mean()):>12.2f}" if has_range else f"{'-':>12}"
        print(
            f"[EVAL] {lv:>5} {pillars:>7.1f} {cnt:>6} {eps:>9.0f} {c:>10.0f} {rate:>9.2%} {f:>6.0f} {mvx:>8.3f} {mcl}"
        )

    env.close()


if __name__ == "__main__":
    main()
