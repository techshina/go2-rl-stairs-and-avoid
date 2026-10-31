"""Evaluate a trained Unitree-Go2-Velocity-Stairs policy on stairs-only terrain.

Spawns envs uniformly across 10 stair-difficulty rows (step height 5cm -> 23cm),
commands a fixed forward velocity, and reports per-difficulty fall rate,
velocity tracking, and stair-clearing over a fixed horizon.

Summit metric: each 8x8 m tile has a 3 m centre platform, a stair band from
1.5 to 3.0 m (Chebyshev distance from the tile origin) and a 1 m flat border.
An episode "summits" when the robot's Chebyshev distance from its tile origin
exceeds 3.2 m — it walked the full staircase (up or down variant alike). The
fall rate alone is biased by whether a policy attempts the climb at all
(a stalling policy falls rarely); summit_rate measures actual climbing.

Usage:
    isaaclab.sh -p src/eval/eval_stairs.py --viz none --num_envs 200 --steps 2000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--steps", type=int, default=2000, help="Control steps to simulate (50 Hz).")
parser.add_argument("--vx", type=float, default=0.5, help="Commanded forward velocity [m/s].")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model .pt (default: latest run).")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity-Stairs", help="Stairs task id to evaluate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from importlib import metadata

import isaaclab.terrains as terrain_gen
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
import tools  # noqa: F401
from tools.agents.rsl_rl_ppo_cfg import AsymmetricPPORunnerCfg
from tools.stairs.velocity_stairs_env_cfg import (
    RobotStairsEnvCfg,
    RobotStairsV2EnvCfg,
)
from tools.stairs_avoid.velocity_heightmap_env_cfg import (
    RobotHeightmapDiverseEnvCfg,
    RobotHeightmapEnvCfg,
)
from tools.stairs_avoid.velocity_extgrid_env_cfg import (
    RobotObstacleExtEnvCfg,
)
from tools.stairs_avoid.velocity_obstacle_env_cfg import (
    RobotObstacleEnvCfg,
)

TASK_CFGS = {
    "Unitree-Go2-Velocity-Stairs": RobotStairsEnvCfg,
    "Unitree-Go2-Velocity-StairsV2": RobotStairsV2EnvCfg,
    # obstacle policy on stairs-only terrain: regression check vs StairsV2
    "Unitree-Go2-Velocity-Obstacle": RobotObstacleEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHM": RobotHeightmapEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHMDiverse": RobotHeightmapDiverseEnvCfg,
    "Unitree-Go2-Velocity-ObstacleExt": RobotObstacleExtEnvCfg,
}

NUM_LEVELS = 10
STEP_H_MIN, STEP_H_MAX = 0.05, 0.23
CLEAR_DIST = 3.2  # [m] Chebyshev distance from tile origin that lies past the last step


def main():
    env_cfg = TASK_CFGS[args_cli.task]()
    env_cfg.scene.num_envs = args_cli.num_envs
    # resolve physics-backend presets by hand (normally done by the hydra
    # pipeline in train.py/play.py, which this script bypasses): PhysX values
    env_cfg.scene.robot.actuators["GO2HV"].armature = 0.0

    # stairs-only terrain: 10 rows ordered easy -> hard (curriculum grid),
    # envs spawn uniformly over all rows and stay there (no level updates)
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
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.5, step_height_range=(STEP_H_MIN, STEP_H_MAX),
                step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.5, step_height_range=(STEP_H_MIN, STEP_H_MAX),
                step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
            ),
        },
    )
    env_cfg.scene.terrain.max_init_terrain_level = None  # spawn across all rows
    env_cfg.curriculum.terrain_levels = None  # freeze level assignment
    env_cfg.curriculum.lin_vel_cmd_levels = None  # freeze command range

    # fixed forward-velocity command
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
    # sub-terrain per env: 0 = pyramid_stairs (spawn on top -> DESCEND),
    # 1 = pyramid_stairs_inv (spawn in pit -> CLIMB); dict order above
    ttypes = env.unwrapped.scene.terrain.terrain_types.clone()
    n = env.unwrapped.num_envs

    falls = torch.zeros(n, device=device)
    timeouts = torch.zeros(n, device=device)
    vx_sum = torch.zeros(n, device=device)
    vx_cnt = torch.zeros(n, device=device)
    origins = env.unwrapped.scene.env_origins[:, :2]  # tile origin per env
    summits = torch.zeros(n, device=device)  # episodes that cleared the stair band
    summits_ok = torch.zeros(n, device=device)  # cleared AND survived to timeout (no fall at all)
    prog_sum = torch.zeros(n, device=device)  # per-episode max Chebyshev progress, summed
    cleared_now = torch.zeros(n, dtype=torch.bool, device=device)
    prog_now = torch.zeros(n, device=device)

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            to = extras["time_outs"].to(device).bool() if "time_outs" in extras else torch.zeros_like(dones).bool()
            d = dones.to(device).bool()
            falls += (d & ~to).float()
            timeouts += (d & to).float()
            # close the episode books first: done envs are already reset, so their
            # current position belongs to the NEXT episode
            summits += (d & cleared_now).float()
            summits_ok += (d & to & cleared_now).float()
            prog_sum += torch.where(d, prog_now, torch.zeros_like(prog_now))
            cleared_now[d] = False
            prog_now[d] = 0.0
            pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
            cheb = (pos - origins).abs().max(dim=1).values
            prog_now = torch.maximum(prog_now, cheb)
            cleared_now |= cheb > CLEAR_DIST
            vx = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, 0]
            vx_sum += vx
            vx_cnt += 1

    print("\n[EVAL] ==== climb vs descend split (summit = cleared the stair band) ====")
    print("[EVAL] summit_ok = cleared the band AND no fall in the episode (strict success)")
    print(f"[EVAL] {'level':>5} {'step_h[cm]':>10} {'dir':>7} {'episodes':>9} {'fall_rate':>9} {'summit_rate':>11} {'summit_ok':>10} {'mean_prog[m]':>12}")
    for lv in range(NUM_LEVELS):
        for tt, tname in ((1, "climb"), (0, "descend")):
            mask = (levels == lv) & (ttypes == tt)
            if int(mask.sum()) == 0:
                continue
            diff = lv / max(NUM_LEVELS - 1, 1)
            step_h = (STEP_H_MIN + (STEP_H_MAX - STEP_H_MIN) * diff) * 100
            f = float(falls[mask].sum()); t = float(timeouts[mask].sum()); eps = f + t
            if eps == 0:
                continue
            s_rate = float(summits[mask].sum()) / eps
            so_rate = float(summits_ok[mask].sum()) / eps
            mprog = float(prog_sum[mask].sum()) / eps
            print(f"[EVAL] {lv:>5} {step_h:>10.1f} {tname:>7} {eps:>9.0f} {f / eps:>9.2%} {s_rate:>11.2%} {so_rate:>10.2%} {mprog:>12.2f}")

    print("\n[EVAL] ==== stairs-only evaluation ====")
    print(f"[EVAL] command vx={args_cli.vx} m/s, horizon={args_cli.steps} steps ({args_cli.steps / 50:.0f}s)")
    print(f"[EVAL] summit = episode's max Chebyshev distance from tile origin > {CLEAR_DIST} m (stairs fully climbed)")
    print(
        f"[EVAL] {'level':>5} {'step_h[cm]':>10} {'#envs':>6} {'episodes':>9} {'falls':>6} {'fall_rate':>9}"
        f" {'summit':>7} {'summit_rate':>11} {'summit_ok':>10} {'mean_prog[m]':>12} {'mean_vx':>8}"
    )
    for lv in range(NUM_LEVELS):
        mask = levels == lv
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        # difficulty of row lv in a curriculum grid: (lv + eps) / num_rows band
        diff = lv / max(NUM_LEVELS - 1, 1)
        step_h = (STEP_H_MIN + (STEP_H_MAX - STEP_H_MIN) * diff) * 100
        f = float(falls[mask].sum())
        t = float(timeouts[mask].sum())
        eps = f + t
        rate = f / eps if eps > 0 else float("nan")
        s = float(summits[mask].sum())
        s_rate = s / eps if eps > 0 else float("nan")
        so_rate = float(summits_ok[mask].sum()) / eps if eps > 0 else float("nan")
        mprog = float(prog_sum[mask].sum()) / eps if eps > 0 else float("nan")
        mvx = float((vx_sum[mask] / vx_cnt[mask]).mean())
        print(
            f"[EVAL] {lv:>5} {step_h:>10.1f} {cnt:>6} {eps:>9.0f} {f:>6.0f} {rate:>9.2%}"
            f" {s:>7.0f} {s_rate:>11.2%} {so_rate:>10.2%} {mprog:>12.2f} {mvx:>8.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()
