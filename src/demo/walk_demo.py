"""Walk a shared Go2 policy checkpoint (.pt) in its own play environment.

For people who have only a model_*.pt file: pick the policy name from
POLICIES below, point --checkpoint at the file, and watch the robot walk
on the terrain the policy was trained for. No training assets are needed.

GUI (DCV / desktop session):
    isaaclab.sh -p src/demo/walk_demo.py --policy stairsavoid-v3 \
        --checkpoint /path/to/model_9999.pt --viz kit

Headless, record a video instead (lands in logs/videos/walk_<policy>/):
    isaaclab.sh -p src/demo/walk_demo.py --policy stairsavoid-v3 \
        --checkpoint /path/to/model_9999.pt --viz none --video

Straight-line walk instead of random commands: add e.g. --vx 0.5

Prerequisites (see README.md):
  * Isaac Sim 6.0.1-rc.7 + Isaac Lab v3.0.0-beta2.patch1 (other versions are
    likely to break the env cfgs; an observation-size error on load is the
    usual symptom)
  * unitree_rl_lab and this repo installed into Isaac Lab's python
  * Go2 USD model available and UNITREE_MODEL_DIR pointing at it
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy",
    type=str,
    required=True,
    choices=["stairs-v1", "stairs-v2", "stairsavoid-v1", "stairsavoid-v2", "stairsavoid-v2d", "stairsavoid-v3"],
    help="Policy name (see POLICIES for the matching task id).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the shared model .pt file.")
parser.add_argument("--num_envs", type=int, default=None, help="Robots to spawn (default: 4, or 1 with --video).")
parser.add_argument("--duration", type=float, default=30.0, help="Walk time in seconds (50 Hz control).")
parser.add_argument("--vx", type=float, default=None, help="Fix the command to straight forward at this speed [m/s].")
parser.add_argument("--video", action="store_true", help="Record an mp4 (use with --viz none).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import sys
import torch

from importlib import metadata

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
import tools  # noqa: F401
from tools.agents.rsl_rl_ppo_cfg import AsymmetricPPORunnerCfg
from tools.stairs.velocity_stairs_env_cfg import (
    RobotStairsPlayEnvCfg,
    RobotStairsV2PlayEnvCfg,
)
from tools.stairs_avoid.velocity_extgrid_env_cfg import (
    RobotObstacleExtPlayEnvCfg,
)
from tools.stairs_avoid.velocity_heightmap_env_cfg import (
    RobotHeightmapDiversePlayEnvCfg,
    RobotHeightmapPlayEnvCfg,
)
from tools.stairs_avoid.velocity_obstacle_env_cfg import (
    RobotObstaclePlayEnvCfg,
)

# policy name -> (gym task id, play env cfg)
POLICIES = {
    "stairs-v1": ("Unitree-Go2-Velocity-Stairs", RobotStairsPlayEnvCfg),
    "stairs-v2": ("Unitree-Go2-Velocity-StairsV2", RobotStairsV2PlayEnvCfg),
    "stairsavoid-v1": ("Unitree-Go2-Velocity-Obstacle", RobotObstaclePlayEnvCfg),
    "stairsavoid-v2": ("Unitree-Go2-Velocity-ObstacleHM", RobotHeightmapPlayEnvCfg),
    "stairsavoid-v2d": ("Unitree-Go2-Velocity-ObstacleHMDiverse", RobotHeightmapDiversePlayEnvCfg),
    "stairsavoid-v3": ("Unitree-Go2-Velocity-ObstacleExt", RobotObstacleExtPlayEnvCfg),
}


def main():
    task_id, cfg_cls = POLICIES[args_cli.policy]
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs or (1 if args_cli.video else 4)
    # resolve physics-backend presets by hand (normally done by the hydra
    # pipeline in train.py/play.py, which this script bypasses)
    env_cfg.scene.robot.actuators["GO2HV"].armature = 0.0

    if args_cli.vx is not None:
        cmd = env_cfg.commands.base_velocity
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)

    agent_cfg = AsymmetricPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    steps = int(args_cli.duration * 50)
    if args_cli.video:
        env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join("logs", "videos", f"walk_{args_cli.policy}"),
            step_trigger=lambda step: step == 0,
            video_length=steps,
            disable_logger=True,
        )
    else:
        env = gym.make(task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[WALK] policy: {args_cli.policy} ({task_id})")
    print(f"[WALK] loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    try:
        runner.load(resume_path)
    except (RuntimeError, KeyError) as err:
        print(f"[WALK][ERROR] checkpoint does not fit the '{args_cli.policy}' network: {err}")
        print("[WALK][ERROR] check that --policy matches the task the .pt was trained on")
        print("[WALK][ERROR] and that Isaac Lab is v3.0.0-beta2.patch1 (observation layout differs across versions).")
        sys.stdout.flush()
        env.close()
        # simulation_app.close() exits the process with code 0, hiding the failure
        os._exit(1)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    update_cam = None
    if args_cli.video:
        # the rgb_array recorder renders a dedicated camera prim whose pose is set
        # only once at construction, so drive it through ViewportManager each step
        from isaacsim.core.rendering_manager import ViewportManager

        cam_prim = env.unwrapped.video_recorder._capture.cfg.camera_prim_path
        anchor = robot.data.root_pos_w[0].tolist()
        alpha = 0.06

        def update_cam():
            pos = robot.data.root_pos_w[0].tolist()
            for i in range(3):
                anchor[i] = (1 - alpha) * anchor[i] + alpha * pos[i]
            ViewportManager.set_camera_view(
                cam_prim,
                eye=[anchor[0] + 1.8, anchor[1] + 1.8, anchor[2] + 0.9],
                target=[anchor[0], anchor[1], anchor[2] + 0.2],
            )

        update_cam()

    falls = 0
    speed_sum = 0.0
    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(steps):
            if not simulation_app.is_running():
                break
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            if update_cam is not None:
                update_cam()
            to = extras["time_outs"].bool() if "time_outs" in extras else torch.zeros_like(dones).bool()
            falls += int((dones.bool() & ~to).sum())
            speed_sum += float(robot.data.root_lin_vel_b[:, :2].norm(dim=1).mean())

    n_steps = t + 1
    print(f"[WALK] done: {n_steps} steps ({n_steps / 50:.0f} s), mean speed {speed_sum / n_steps:.2f} m/s, falls {falls}")
    if args_cli.video:
        print(f"[WALK] video: logs/videos/walk_{args_cli.policy}/")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
