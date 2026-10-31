"""Record a single-robot (num_envs=1) demo video with a smoothed follow camera.

Modes:
  stairs   -- inverted-pyramid stairs at a fixed step height (robot climbs out).
              --tile 8 (default) matches the eval_stairs.py geometry (~5 steps);
              --tile 14 gives a long staircase (~15 steps).
  obstacle -- box-pillar field at a fixed count (robot weaves through).

--task switches the policy: the Obstacle task (default) or StairsV2, which has
no horizontal lidar and runs the pillar course on height_scan alone.
--terrain_seed pins the layout so two policies can be compared on identical terrain.

Usage (from the repo root; videos land in logs/videos/<out>/):
    isaaclab.sh -p src/demo/record_demo.py --viz none --mode stairs --step_h 0.13 --out stairs_13cm
    isaaclab.sh -p src/demo/record_demo.py --viz none --mode stairs --step_h 0.13 --tile 14 --out stairs_13cm_long
    isaaclab.sh -p src/demo/record_demo.py --viz none --mode obstacle --pillars 60 --terrain_seed 42 --out obstacle
    isaaclab.sh -p src/demo/record_demo.py --viz none --task Unitree-Go2-Velocity-StairsV2 \\
        --mode obstacle --pillars 60 --terrain_seed 42 --out obstacle_stairsv2

Headless note: the rgb_array recorder renders a dedicated camera prim whose pose is
applied once at construction, so the follow camera drives that prim directly via
ViewportManager.set_camera_view every control step (sim.set_camera_view has no effect).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["stairs", "obstacle"], required=True)
parser.add_argument("--step_h", type=float, default=0.13, help="Stair step height [m].")
parser.add_argument("--pillars", type=int, default=10, help="Pillar count (obstacle mode).")
parser.add_argument("--pillar_size", type=float, default=0.5, help="Pillar footprint [m].")
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--video_length", type=int, default=1000, help="Control steps (50 Hz).")
parser.add_argument("--tile", type=float, default=8.0, help="Terrain tile size [m] (stairs mode).")
parser.add_argument(
    "--task", type=str, default="Unitree-Go2-Velocity-Obstacle",
    choices=["Unitree-Go2-Velocity-Obstacle", "Unitree-Go2-Velocity-ObstacleHM", "Unitree-Go2-Velocity-ObstacleHMDiverse", "Unitree-Go2-Velocity-StairsV2", "Unitree-Go2-Velocity-ObstacleExt"],
    help="Policy/task to record (StairsV2 on obstacle terrain runs blind — no lidar).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model .pt (default: latest run).")
parser.add_argument("--terrain_seed", type=int, default=None,
                    help="Terrain generator seed (same seed -> same pillar/stair layout).")
parser.add_argument("--show_scan", action="store_true", default=False,
                    help="Draw the height-scan hit points (coarse grid red, fine grid green).")
parser.add_argument("--out", type=str, required=True, help="Output folder name (under logs/videos/).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

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
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import PhysxEventCfg
from tools.stairs.velocity_stairs_env_cfg import RobotStairsV2EnvCfg
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
    "Unitree-Go2-Velocity-Obstacle": RobotObstacleEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHM": RobotHeightmapEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHMDiverse": RobotHeightmapDiverseEnvCfg,
    "Unitree-Go2-Velocity-StairsV2": RobotStairsV2EnvCfg,
    "Unitree-Go2-Velocity-ObstacleExt": RobotObstacleExtEnvCfg,
}
OUT_ROOT = os.path.abspath(os.path.join("logs", "videos"))


def main():
    env_cfg = TASK_CFGS[args_cli.task]()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.actuators["GO2HV"].armature = 0.0  # physics preset, as in eval_stairs.py

    if args_cli.mode == "stairs":
        # default 8 m tile / 3 m platform = the eval_stairs.py geometry (~5 steps);
        # --tile 14 gives a long staircase (the lidar-gated Obstacle task stalls in front of it)
        tile = args_cli.tile
        sub = {
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(args_cli.step_h, args_cli.step_h),
                step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
            ),
        }
        # follow-cam offset: side-on view of the climb (+x)
        cam_offset, look_offset = (0.6, -3.6, 1.5), (0.6, 0.0, 0.2)
    else:
        # 16 m tile so the robot stays inside the pillar field for the whole clip
        tile = 16.0
        n, s = args_cli.pillars, args_cli.pillar_size
        sub = {
            "obstacles": terrain_gen.MeshRepeatedBoxesTerrainCfg(
                proportion=1.0,
                object_type="box",
                object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=n, height=1.5, size=(s, s)
                ),
                object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=n, height=1.5, size=(s, s)
                ),
                abs_height_noise=(0.0, 0.0),
                platform_width=1.5,
                platform_height=0.0,
            ),
        }
        # follow-cam offset: high three-quarter view from behind-left
        cam_offset, look_offset = (-3.2, -3.2, 2.4), (0.5, 0.0, 0.2)

    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(tile, tile), border_width=20.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        difficulty_range=(1.0, 1.0), use_cache=False, curriculum=True,
        sub_terrains=sub, seed=args_cli.terrain_seed,
    )
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None

    # deterministic straight-ahead run from the tile center
    cmd = env_cfg.commands.base_velocity
    cmd.rel_standing_envs = 0.0
    cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.debug_vis = False  # hide the floating command-arrow marker
    # resolve the events preset by hand (this script bypasses hydra) and make
    # the spawn deterministic: tile center, facing +x, no random pushes
    events = PhysxEventCfg()
    events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    events.push_robot = None
    env_cfg.events = events

    if args_cli.show_scan:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkersCfg

        env_cfg.scene.height_scanner.debug_vis = True
        env_cfg.scene.height_scanner.visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/FineScan",
            markers={"hit": sim_utils.SphereCfg(
                radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.1)))},
        )
        if hasattr(env_cfg.scene, "coarse_scanner"):
            env_cfg.scene.coarse_scanner.debug_vis = True
            env_cfg.scene.coarse_scanner.visualizer_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/CoarseScan",
                markers={"hit": sim_utils.SphereCfg(
                    radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)))},
            )

    env_cfg.viewer.eye = cam_offset
    env_cfg.viewer.lookat = look_offset

    agent_cfg = AsymmetricPPORunnerCfg()
    agent_cfg.experiment_name = args_cli.task.lower().replace("-", "_")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(OUT_ROOT, args_cli.out),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[REC] Loading checkpoint: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # smoothed follow camera. The rgb_array recorder renders a dedicated camera
    # prim whose pose is set only once at construction (see IsaacsimKitPerspectiveVideo),
    # so drive that prim directly through ViewportManager every control step.
    from isaacsim.core.rendering_manager import ViewportManager

    robot = env.unwrapped.scene["robot"]
    cam_prim = env.unwrapped.video_recorder._capture.cfg.camera_prim_path
    anchor = robot.data.root_pos_w[0].tolist()
    alpha = 0.06

    def update_cam():
        pos = robot.data.root_pos_w[0].tolist()
        for i in range(3):
            anchor[i] = (1 - alpha) * anchor[i] + alpha * pos[i]
        ViewportManager.set_camera_view(
            cam_prim,
            eye=[anchor[0] + cam_offset[0], anchor[1] + cam_offset[1], anchor[2] + cam_offset[2]],
            target=[anchor[0] + look_offset[0], anchor[1] + look_offset[1], anchor[2] + look_offset[2]],
        )

    # with --viz none no visualizer dispatches the marker-update callbacks, so
    # the scan hit-point markers must be driven manually each step
    def update_markers():
        if args_cli.show_scan:
            env.unwrapped.sim.vis_marker_registry.dispatch_callbacks()

    update_cam()
    update_markers()
    obs = env.get_observations()
    speeds = []
    with torch.inference_mode():
        for t in range(args_cli.video_length):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            speeds.append(float(robot.data.root_lin_vel_b[0, :2].norm()))
            update_cam()
            update_markers()
            if bool(dones.any()) and t < args_cli.video_length - 1:
                print(f"[REC] episode reset at step {t}")
    sp = torch.tensor(speeds)
    print(
        f"[REC] speed stats: mean {sp.mean():.3f} m/s, stopped(<0.1) {(sp < 0.1).float().mean():.1%},"
        f" slow(<0.25) {(sp < 0.25).float().mean():.1%} of {len(sp)} steps"
    )
    env.close()
    print(f"[REC] done: {os.path.join(OUT_ROOT, args_cli.out)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
