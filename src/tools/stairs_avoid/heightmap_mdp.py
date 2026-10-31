"""MDP terms for the height-map variant of the stairs + obstacle-avoidance task.

Replaces the horizontal 2D lidar with a coarse, forward-biased elevation grid
(``coarse_scanner`` in the scene). Obstacle *cells* are classified from the
clean (privileged) grid inside the reward terms only — the policy sees raw
heights, so no threshold logic has to be reproduced on the robot.

A cell is an obstacle when BOTH hold:

* absolute: it stands more than ``h_abs`` above the ground under the robot
  (a single stair step never does; a pillar or box always does), and
* local step: it rises more than ``h_step`` above the lowest of its 4-neighbours
  (a staircase climbs at most one riser per cell, so distant high steps are
  excluded here while the near ones are excluded by the absolute test).

Grid layout: ``GridPatternCfg(ordering="xy")`` flattens with x as the inner
loop, so the flat ray vector reshapes to ``(N, ny, nx)``.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Coarse grid geometry. Must match ``coarse_scanner`` in velocity_heightmap_env_cfg.py.
COARSE_RES = 0.25
COARSE_SIZE = (5.0, 3.0)  # (x length, y width) [m]
COARSE_X_OFFSET = 2.0  # grid centre ahead of base [m] -> covers x in [-0.5, 4.5]
COARSE_NX = int(round(COARSE_SIZE[0] / COARSE_RES)) + 1  # 21
COARSE_NY = int(round(COARSE_SIZE[1] / COARSE_RES)) + 1  # 13
COARSE_MAX_RANGE = 5.0  # "no obstacle" distance for gating


def _grid_xy(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Base-frame (x, y) of every coarse cell, shape (ny, nx). Cached per device."""
    key = str(device)
    if key not in _grid_xy.cache:
        xs = torch.arange(-COARSE_SIZE[0] / 2, COARSE_SIZE[0] / 2 + 1e-9, COARSE_RES, device=device) + COARSE_X_OFFSET
        ys = torch.arange(-COARSE_SIZE[1] / 2, COARSE_SIZE[1] / 2 + 1e-9, COARSE_RES, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
        _grid_xy.cache[key] = (gx, gy)
    return _grid_xy.cache[key]


_grid_xy.cache = {}


def coarse_heights(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terrain height of every coarse cell, world z, shape (N, ny, nx). Misses -> NaN."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ray_hits_w.torch[..., 2].view(env.num_envs, COARSE_NY, COARSE_NX)


def ground_height_under_robot(env: ManagerBasedRLEnv, fine_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ground z under the robot: median of the fine height_scanner hits (robust to a foot on a step edge)."""
    sensor = env.scene.sensors[fine_sensor_cfg.name]
    z = sensor.data.ray_hits_w.torch[..., 2]
    return torch.nan_to_num(z, nan=0.0).median(dim=1).values


def obstacle_cells(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    fine_sensor_cfg: SceneEntityCfg,
    h_abs: float = 0.35,
    h_step: float = 0.35,
) -> torch.Tensor:
    """Boolean obstacle mask over the coarse grid, shape (N, ny, nx)."""
    h = coarse_heights(env, sensor_cfg)
    ground = ground_height_under_robot(env, fine_sensor_cfg)
    # rays that miss (off the terrain mesh) read as ground so they never count as obstacles
    h = torch.where(torch.isnan(h), ground[:, None, None].expand_as(h), h)
    abs_ok = (h - ground[:, None, None]) > h_abs
    hp = F.pad(h.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    nbr_min = torch.minimum(
        torch.minimum(hp[:, :-2, 1:-1], hp[:, 2:, 1:-1]),
        torch.minimum(hp[:, 1:-1, :-2], hp[:, 1:-1, 2:]),
    )
    step_ok = (h - nbr_min) > h_step
    return abs_ok & step_ok


def _cell_dist_yaw(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gx, gy = _grid_xy(device)
    return torch.sqrt(gx**2 + gy**2), torch.atan2(gy, gx)


def coarse_height_scan(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5, clip_range: float = 2.0
) -> torch.Tensor:
    """Observation: coarse elevation relative to the sensor, flat (N, ny*nx), misses -> +clip (far below)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    rel = sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset
    rel = torch.nan_to_num(rel, nan=clip_range, posinf=clip_range, neginf=-clip_range)
    return rel.clamp(-clip_range, clip_range)


def heightmap_obstacle_proximity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    fine_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.8,
    h_abs: float = 0.35,
    h_step: float = 0.35,
) -> torch.Tensor:
    """Quadratic penalty on the distance to the nearest obstacle cell (0 when none within ``threshold``)."""
    mask = obstacle_cells(env, sensor_cfg, fine_sensor_cfg, h_abs, h_step)
    dist, _ = _cell_dist_yaw(mask.device)
    d_min = torch.where(mask, dist.unsqueeze(0), torch.full_like(mask, COARSE_MAX_RANGE, dtype=dist.dtype))
    d_min = d_min.flatten(1).min(dim=1).values
    return ((threshold - d_min).clamp(min=0.0) / threshold) ** 2


def track_lin_vel_xy_heightmap_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    fine_sensor_cfg: SceneEntityCfg,
    gate_distance: float = 1.0,
    cone_half_angle: float = 1.05,
    h_abs: float = 0.35,
    h_step: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """xy-velocity tracking faded by the clearance to obstacle cells in the commanded direction.

    Same shape as the lidar version (``obstacle_mdp.track_lin_vel_xy_obstacle_gated``):
    gate = clamp(d_min / gate_distance, 0, 1), d_min over obstacle cells inside a cone
    around the commanded heading. Stair cells are never obstacles, so climbing is
    never gated. Near-zero (standing) commands are never gated.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    track = torch.exp(-lin_vel_error / std**2)

    mask = obstacle_cells(env, sensor_cfg, fine_sensor_cfg, h_abs, h_step)
    dist, yaw = _cell_dist_yaw(mask.device)
    # the grid is yaw-aligned with the base, so cell yaw is already in the base frame
    cmd_dir_b = torch.atan2(cmd[:, 1], cmd[:, 0])
    in_cone = math_utils.wrap_to_pi(yaw.unsqueeze(0) - cmd_dir_b[:, None, None]).abs() < cone_half_angle
    d_min = torch.where(mask & in_cone, dist.unsqueeze(0), torch.full_like(mask, COARSE_MAX_RANGE, dtype=dist.dtype))
    d_min = d_min.flatten(1).min(dim=1).values
    gate = (d_min / gate_distance).clamp(0.0, 1.0)
    gate = torch.where(torch.norm(cmd, dim=1) < 0.1, torch.ones_like(gate), gate)
    return track * gate
