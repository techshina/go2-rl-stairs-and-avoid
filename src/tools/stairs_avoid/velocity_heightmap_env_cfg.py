"""Stairs + obstacle avoidance (single policy), height-map variant.

Variant of :class:`RobotObstacleEnvCfg`. The horizontal 2D lidar cannot
not tell a long staircase's risers from a pillar (both are "something 1.6 m
ahead" in a single scan plane), so its reward gate stalls the robot in front
of long stairs. This variant replaces the lidar with a coarse, forward-biased
elevation grid and classifies obstacle cells by height (absolute AND local
step), so stairs are never gated.

Relative to :class:`RobotObstacleEnvCfg`:

* sensor: ``lidar_scanner`` removed; ``coarse_scanner`` added
  (5.0 x 3.0 m grid, 0.25 m, centred 2.0 m ahead -> x in [-0.5, 4.5], 21 x 13 = 273 rays),
* observation: ``lidar_scan`` (72) replaced by ``coarse_height_scan`` (273) in actor and critic,
  actor 1 frame = 45 + 187 + 273 = 505, x5 history = 2,525; critic = 247 + 273 = 520,
* rewards: gated tracking and proximity penalty consume obstacle cells instead of
  lidar rays (same weights, gate_distance 1.0 m, threshold 0.8 m), ``head_contacts`` unchanged,
* terrain, terminations, curriculum, network: unchanged (comparable with StairsV2 / the lidar task).

Real-robot note: both grids come from the same L1-lidar elevation map, cut at
two resolutions — no horizontal scan extraction needed.
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RewardsCfg, RobotSceneCfg
from tools.stairs.velocity_stairs_env_cfg import (
    RobotStairsV2EnvCfg,
    StairsObservationsCfg,
)

from . import heightmap_mdp
from .velocity_obstacle_env_cfg import OBSTACLE_TERRAINS_CFG

H_ABS = 0.35
H_STEP = 0.35


@configclass
class HeightmapSceneCfg(RobotSceneCfg):
    """Adds the coarse forward-biased elevation grid to the stairs scene."""

    coarse_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        # same 20 m drop-cast as the fine height_scanner; centre shifted 2 m ahead
        offset=RayCasterCfg.OffsetCfg(pos=(heightmap_mdp.COARSE_X_OFFSET, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=heightmap_mdp.COARSE_RES, size=list(heightmap_mdp.COARSE_SIZE), ordering="xy"
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class HeightmapObservationsCfg(StairsObservationsCfg):
    """Stairs observations + coarse elevation grid."""

    @configclass
    class PolicyCfg(StairsObservationsCfg.PolicyCfg):
        # same +-10 cm noise as the fine scan (stand-in for a real elevation map)
        coarse_height_scan = ObsTerm(
            func=heightmap_mdp.coarse_height_scan,
            params={"sensor_cfg": SceneEntityCfg("coarse_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-2.0, 2.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(StairsObservationsCfg.CriticCfg):
        coarse_height_scan = ObsTerm(
            func=heightmap_mdp.coarse_height_scan,
            params={"sensor_cfg": SceneEntityCfg("coarse_scanner")},
            clip=(-2.0, 2.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class HeightmapRewardsCfg(RewardsCfg):
    """StairsV2 rewards with height-map-gated tracking and an obstacle-cell proximity penalty."""

    track_lin_vel_xy = RewTerm(
        func=heightmap_mdp.track_lin_vel_xy_heightmap_gated,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": SceneEntityCfg("coarse_scanner"),
            "fine_sensor_cfg": SceneEntityCfg("height_scanner"),
            "gate_distance": 1.0,
            "cone_half_angle": 1.05,  # ~60 deg
            "h_abs": H_ABS,
            "h_step": H_STEP,
        },
    )
    obstacle_proximity = RewTerm(
        func=heightmap_mdp.heightmap_obstacle_proximity,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("coarse_scanner"),
            "fine_sensor_cfg": SceneEntityCfg("height_scanner"),
            "threshold": 0.8,
            "h_abs": H_ABS,
            "h_step": H_STEP,
        },
    )
    head_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )


@configclass
class RobotHeightmapEnvCfg(RobotStairsV2EnvCfg):
    """StairsV2 + obstacle avoidance from a coarse elevation grid (no horizontal lidar)."""

    scene: HeightmapSceneCfg = HeightmapSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HeightmapObservationsCfg = HeightmapObservationsCfg()
    rewards: HeightmapRewardsCfg = HeightmapRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # the stairs post-init installs STAIRS_TERRAINS_CFG unconditionally,
        # so the obstacle mix (and its curriculum flag) must be set after it
        self.scene.terrain.terrain_generator = OBSTACLE_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.coarse_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RobotHeightmapPlayEnvCfg(RobotHeightmapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# StairsV2 terrain mix + diverse obstacle classes: boxes of varied
# height/footprint, axis-aligned wall segments (long thin boxes, one class per
# axis since repeated objects get no yaw randomization), thick cylinders
# (poles/trees/legs) and cones. Objects sit half-buried (z=0 centred), so the
# effective height above ground is cfg_height/2 + noise/2. Every class keeps
# its effective height >= 0.4 m — above the H_ABS/H_STEP classifier threshold
# (0.35 m) — so no "maybe climbable" heights put the classifier at odds with
# the physics. Cylinder/cone radii stay >= 0.2 m: the farthest a disk centre can sit from a
# 0.25 m grid point is 0.177 m, so radius >= 0.18 guarantees at least one ray
# hit (radius 0.15 slips through the grid). The tall-pillar class is kept for
# comparability with the Obstacle / ObstacleHM tasks. Raw proportions total 0.33 on top of the
# 1.0 stairs mix -> obstacles occupy ~25% of terrain columns.
DIVERSE_OBSTACLES_TERRAINS_CFG = OBSTACLE_TERRAINS_CFG.copy()
DIVERSE_OBSTACLES_TERRAINS_CFG.sub_terrains["obstacles"].proportion = 0.06  # tall pillars, as in the Obstacle task
DIVERSE_OBSTACLES_TERRAINS_CFG.sub_terrains["boxes_mixed"] = terrain_gen.MeshRepeatedBoxesTerrainCfg(
    proportion=0.08,
    object_type="box",
    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=4, height=1.6, size=(0.4, 0.4)
    ),
    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=10, height=1.6, size=(0.9, 0.9)
    ),
    abs_height_noise=(-0.8, 0.8),  # effective 0.4..1.2 m
    platform_width=1.5,
    platform_height=0.0,
)
DIVERSE_OBSTACLES_TERRAINS_CFG.sub_terrains["walls_x"] = terrain_gen.MeshRepeatedBoxesTerrainCfg(
    proportion=0.05,
    object_type="box",
    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=2, height=1.6, size=(1.2, 0.3)
    ),
    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=4, height=1.6, size=(2.5, 0.4)
    ),
    abs_height_noise=(-0.6, 0.4),  # effective 0.5..1.0 m
    platform_width=1.5,
    platform_height=0.0,
)
DIVERSE_OBSTACLES_TERRAINS_CFG.sub_terrains["walls_y"] = terrain_gen.MeshRepeatedBoxesTerrainCfg(
    proportion=0.05,
    object_type="box",
    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=2, height=1.6, size=(0.3, 1.2)
    ),
    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=4, height=1.6, size=(0.4, 2.5)
    ),
    abs_height_noise=(-0.6, 0.4),  # effective 0.5..1.0 m
    platform_width=1.5,
    platform_height=0.0,
)
DIVERSE_OBSTACLES_TERRAINS_CFG.sub_terrains["cylinders"] = terrain_gen.MeshRepeatedCylindersTerrainCfg(
    proportion=0.05,
    object_type="cylinder",
    object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
        num_objects=4, height=1.6, radius=0.2
    ),
    object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
        num_objects=10, height=1.6, radius=0.35
    ),
    abs_height_noise=(-0.8, 0.8),  # effective 0.4..1.2 m
    platform_width=1.5,
    platform_height=0.0,
)
DIVERSE_OBSTACLES_TERRAINS_CFG.sub_terrains["cones"] = terrain_gen.MeshRepeatedPyramidsTerrainCfg(
    proportion=0.04,
    object_type="cone",
    object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        num_objects=4, height=1.2, radius=0.25
    ),
    object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
        num_objects=8, height=1.6, radius=0.35
    ),
    abs_height_noise=(-0.2, 0.2),  # effective 0.5..0.9 m
    platform_width=1.5,
    platform_height=0.0,
)


@configclass
class RobotHeightmapDiverseEnvCfg(RobotHeightmapEnvCfg):
    """Height-map task with diverse obstacle classes: pillars, boxes, walls, cylinders, cones."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = DIVERSE_OBSTACLES_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True


@configclass
class RobotHeightmapDiversePlayEnvCfg(RobotHeightmapDiverseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
