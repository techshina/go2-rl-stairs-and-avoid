"""MDP terms for the extended-fine-grid task (``Unitree-Go2-Velocity-ObstacleExt``).

In the height-map variant the stair skill itself is intact — clamping the
coarse channel to a flat constant restores climbing to specialist level
(15 cm summit 53% -> 89%) — so the coarse channel perturbs precise foot
placement at runtime. This task removes the coarse
grid entirely and instead EXTENDS the fine body grid forward:

* one scanner: x in [-0.8, +2.0] m x y in [-1.0, +1.0] m at 0.1 m -> 29 x 21 = 609 rays
  (sensor offset +0.6 m so the pattern is centred, yaw-aligned, drop-cast),
* actor sees it as two terms: the NEAR block — the original 17 x 11 sub-grid in
  the specialist's exact cell order, x5 history (bit-compatible with the
  stairs-specialist weights) — and the FAR extension (422 cells, no history),
* rewards run the verified 2-condition obstacle-cell classifier on this grid
  (gate 1.0 m / proximity 0.8 m as in the coarse variant).

Actor: 45x5 + 187x5 + 422 = 1,582. Critic: 60 + 609 = 669.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Extended grid geometry. Must match ``height_scanner`` in velocity_extgrid_env_cfg.py.
EXT_RES = 0.1
EXT_SIZE = (2.8, 2.0)  # pattern (x length, y width) [m]
EXT_X_OFFSET = 0.6  # sensor offset ahead of base -> base-frame x in [-0.8, +2.0]
EXT_NX = int(round(EXT_SIZE[0] / EXT_RES)) + 1  # 29
EXT_NY = int(round(EXT_SIZE[1] / EXT_RES)) + 1  # 21
EXT_MAX_RANGE = 2.5  # "no obstacle" distance for gating

# Near sub-grid = the stock fine grid (x -0.8..0.8, y -0.5..0.5): 17 x 11 = 187 cells.
# Flat ray order is x-fast (GridPatternCfg ordering "xy"): idx = iy * EXT_NX + ix.
NEAR_NX, NEAR_NY = 17, 11
_near, _far = [], []
_near_set = set()
for _iyp in range(NEAR_NY):
    for _ixp in range(NEAR_NX):
        _idx = (_iyp + 5) * EXT_NX + _ixp  # near y starts 5 rows in; near x starts at column 0
        _near.append(_idx)
        _near_set.add(_idx)
for _idx in range(EXT_NX * EXT_NY):
    if _idx not in _near_set:
        _far.append(_idx)
NEAR_IDX = tuple(_near)  # length 187, in the stock scanner's own flat order
FAR_IDX = tuple(_far)  # length 422


def _idx_tensor(env: ManagerBasedRLEnv, name: str, values: tuple[int, ...]) -> torch.Tensor:
    cache = _idx_tensor.cache.setdefault(str(env.device), {})
    if name not in cache:
        cache[name] = torch.tensor(values, device=env.device, dtype=torch.long)
    return cache[name]


_idx_tensor.cache = {}


def _rel_heights(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float) -> torch.Tensor:
    """sensor_z - hit_z - offset for every ray, shape (N, 609). NaN -> 0."""
    sensor = env.scene.sensors[sensor_cfg.name]
    rel = sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset
    return torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)


def ext_height_scan_near(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """The stock 17x11 body grid extracted from the extended scanner, in the stock cell order.

    Values and layout are identical to ``mdp.height_scan`` on the original fine
    scanner (same z drop height, same offset), so specialist weights transfer
    bit-exactly onto this block.
    """
    return _rel_heights(env, sensor_cfg, offset).index_select(1, _idx_tensor(env, "near", NEAR_IDX))


def ext_height_scan_far(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """The 422 extension cells (forward to 2.0 m, sideways to +-1.0 m)."""
    return _rel_heights(env, sensor_cfg, offset).index_select(1, _idx_tensor(env, "far", FAR_IDX))


def _ext_dist_yaw(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    if key not in _ext_dist_yaw.cache:
        xs = torch.arange(-EXT_SIZE[0] / 2, EXT_SIZE[0] / 2 + 1e-9, EXT_RES, device=device) + EXT_X_OFFSET
        ys = torch.arange(-EXT_SIZE[1] / 2, EXT_SIZE[1] / 2 + 1e-9, EXT_RES, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
        _ext_dist_yaw.cache[key] = (torch.sqrt(gx**2 + gy**2), torch.atan2(gy, gx))
    return _ext_dist_yaw.cache[key]


_ext_dist_yaw.cache = {}


def ext_obstacle_cells(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    h_abs: float = 0.35,
    h_step: float = 0.35,
) -> torch.Tensor:
    """Boolean obstacle mask over the extended grid, shape (N, ny, nx).

    Foot-reference height = median of the NEAR sub-grid (same robust reference
    as the specialist uses), not of the whole grid — a long staircase ahead
    must not drag the reference up.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    h_flat = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(h_flat.index_select(1, _idx_tensor(env, "near", NEAR_IDX)), nan=0.0).median(dim=1).values
    h = h_flat.view(env.num_envs, EXT_NY, EXT_NX)
    h = torch.where(torch.isnan(h), ground[:, None, None].expand_as(h), h)
    abs_ok = (h - ground[:, None, None]) > h_abs
    hp = F.pad(h.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    nbr_min = torch.minimum(
        torch.minimum(hp[:, :-2, 1:-1], hp[:, 2:, 1:-1]),
        torch.minimum(hp[:, 1:-1, :-2], hp[:, 1:-1, 2:]),
    )
    return abs_ok & ((h - nbr_min) > h_step)


def ext_obstacle_proximity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.8,
    h_abs: float = 0.35,
    h_step: float = 0.35,
) -> torch.Tensor:
    """Quadratic penalty on the distance to the nearest obstacle cell."""
    mask = ext_obstacle_cells(env, sensor_cfg, h_abs, h_step)
    dist, _ = _ext_dist_yaw(mask.device)
    d_min = torch.where(mask, dist.unsqueeze(0), torch.full_like(mask, EXT_MAX_RANGE, dtype=dist.dtype))
    d_min = d_min.flatten(1).min(dim=1).values
    return ((threshold - d_min).clamp(min=0.0) / threshold) ** 2


def track_lin_vel_xy_ext_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    gate_distance: float = 1.0,
    cone_half_angle: float = 1.05,
    h_abs: float = 0.35,
    h_step: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """xy-velocity tracking faded by the clearance to obstacle cells in the commanded direction."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    track = torch.exp(-lin_vel_error / std**2)

    mask = ext_obstacle_cells(env, sensor_cfg, h_abs, h_step)
    dist, yaw = _ext_dist_yaw(mask.device)
    cmd_dir_b = torch.atan2(cmd[:, 1], cmd[:, 0])
    in_cone = math_utils.wrap_to_pi(yaw.unsqueeze(0) - cmd_dir_b[:, None, None]).abs() < cone_half_angle
    d_min = torch.where(mask & in_cone, dist.unsqueeze(0), torch.full_like(mask, EXT_MAX_RANGE, dtype=dist.dtype))
    d_min = d_min.flatten(1).min(dim=1).values
    gate = (d_min / gate_distance).clamp(0.0, 1.0)
    gate = torch.where(torch.norm(cmd, dim=1) < 0.1, torch.ones_like(gate), gate)
    return track * gate
