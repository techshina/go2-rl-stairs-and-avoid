"""Export a trained policy checkpoint to TorchScript.

Loads the rsl_rl checkpoint through its task env (dims must match), exports a
self-contained .jit policy (torch-only, no rsl_rl / Isaac Lab needed to run)
and verifies the export against the runner's inference policy on random inputs.

Usage:
    isaaclab.sh -p src/utils/export_policy_jit.py --viz none \
        --task Unitree-Go2-Velocity-ObstacleExt \
        --checkpoint logs/rsl_rl/.../model_9999.pt \
        --out policy.jit
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="Output .jit path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from importlib import metadata

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
import tools  # noqa: F401
from tools.agents.rsl_rl_ppo_cfg import AsymmetricPPORunnerCfg
from tools.stairs.velocity_stairs_env_cfg import (
    RobotStairsEnvCfg,
    RobotStairsV2EnvCfg,
)
from tools.stairs_avoid.velocity_extgrid_env_cfg import RobotObstacleExtEnvCfg
from tools.stairs_avoid.velocity_heightmap_env_cfg import (
    RobotHeightmapDiverseEnvCfg,
    RobotHeightmapEnvCfg,
)
from tools.stairs_avoid.velocity_obstacle_env_cfg import RobotObstacleEnvCfg

TASK_CFGS = {
    "Unitree-Go2-Velocity-Stairs": RobotStairsEnvCfg,
    "Unitree-Go2-Velocity-StairsV2": RobotStairsV2EnvCfg,
    "Unitree-Go2-Velocity-Obstacle": RobotObstacleEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHM": RobotHeightmapEnvCfg,
    "Unitree-Go2-Velocity-ObstacleHMDiverse": RobotHeightmapDiverseEnvCfg,
    "Unitree-Go2-Velocity-ObstacleExt": RobotObstacleExtEnvCfg,
}


def main():
    env_cfg = TASK_CFGS[args_cli.task]()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.actuators["GO2HV"].armature = 0.0

    agent_cfg = AsymmetricPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[EXPORT] loading: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    out_dir = os.path.dirname(os.path.abspath(args_cli.out)) or "."
    out_name = os.path.basename(args_cli.out)
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=out_dir, filename=out_name)

    # verify: jit output must match the runner's inference policy
    num_obs = env.get_observations().shape[1]
    jit_policy = torch.jit.load(os.path.join(out_dir, out_name)).to(env.unwrapped.device)
    obs = torch.randn(64, num_obs, device=env.unwrapped.device)
    with torch.inference_mode():
        diff = (jit_policy(obs) - policy(obs)).abs().max().item()
    print(f"[EXPORT] wrote {os.path.join(out_dir, out_name)} (obs dim {num_obs})")
    print(f"[EXPORT] max |jit - runner| over 64 random obs: {diff:.3e}")
    if diff > 1e-5:
        raise SystemExit(f"[EXPORT][ERROR] export mismatch: {diff}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
