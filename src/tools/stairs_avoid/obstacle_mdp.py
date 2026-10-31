"""MDP terms for the Go2 stairs + obstacle-avoidance task (single policy).

All terms consume a horizontal 2D-lidar :class:`RayCaster` (single channel,
yaw-aligned) named ``lidar_scanner`` in the scene. Missed rays are mapped to
``max_range`` so open space always reads as "far".
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _lidar_distances(sensor, max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ray hit vectors (world frame) and distances, clamped to ``max_range``."""
    hits = sensor.data.ray_hits_w.torch  # (N, R, 3)
    pos = sensor.data.pos_w.torch.unsqueeze(1)  # (N, 1, 3)
    vec = hits - pos
    dist = torch.norm(vec, dim=-1)
    dist = torch.nan_to_num(dist, nan=max_range, posinf=max_range, neginf=max_range)
    return vec, dist.clamp(0.0, max_range)


def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_range: float = 5.0) -> torch.Tensor:
    """Normalized lidar ranges in [0, 1] (1 = free space at/beyond ``max_range``)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    _, dist = _lidar_distances(sensor, max_range)
    return dist / max_range


def lidar_obstacle_proximity(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.8, max_range: float = 5.0
) -> torch.Tensor:
    """Quadratic penalty for obstacles inside ``threshold`` (mean over rays).

    The lidar plane sits ~0.7 m above ground, so stair risers only enter it
    at >0.9 m ahead even on the steepest (23 cm) stairs — with ``threshold``
    below that, stair climbing is never penalized, only true obstacles.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    _, dist = _lidar_distances(sensor, max_range)
    proximity = ((threshold - dist).clamp(min=0.0) / threshold) ** 2
    return proximity.mean(dim=1)


def track_lin_vel_xy_obstacle_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    gate_distance: float = 1.0,
    cone_half_angle: float = 1.05,
    max_range: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity-tracking reward that fades out when the commanded direction is blocked.

    Standard exponential xy-velocity tracking, multiplied by a clearance gate:
    the minimum lidar range inside a cone of ``cone_half_angle`` around the
    commanded direction, normalized by ``gate_distance``. With an obstacle
    directly in the commanded path the gate goes to 0, so deviating from the
    command to swerve around it costs nothing — this is what licenses the
    policy to ignore the command near obstacles. Near-zero (standing)
    commands are never gated.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    track = torch.exp(-lin_vel_error / std**2)

    sensor = env.scene.sensors[sensor_cfg.name]
    vec, dist = _lidar_distances(sensor, max_range)
    # commanded direction in world yaw = base yaw + command heading (base frame)
    _, _, base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    cmd_dir_w = base_yaw + torch.atan2(cmd[:, 1], cmd[:, 0])
    ray_yaw = torch.atan2(vec[..., 1], vec[..., 0])
    in_cone = math_utils.wrap_to_pi(ray_yaw - cmd_dir_w.unsqueeze(1)).abs() < cone_half_angle
    d_min = torch.where(in_cone, dist, torch.full_like(dist, max_range)).min(dim=1).values
    gate = (d_min / gate_distance).clamp(0.0, 1.0)
    gate = torch.where(torch.norm(cmd, dim=1) < 0.1, torch.ones_like(gate), gate)
    return track * gate
