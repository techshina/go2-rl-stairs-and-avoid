"""Stairs + obstacle avoidance in a single policy, on top of the StairsV2 task.

Extends :class:`RobotStairsV2EnvCfg` (design goal: a single policy that
walks stairs AND avoids obstacles — no hierarchical nav layer):

* adds an ``obstacles`` sub-terrain (tall box pillars, curriculum scales
  density/size) to the StairsV2 terrain mix,
* adds a horizontal 2D-lidar ray-cast observation (72 rays, yaw-aligned,
  ~0.7 m above ground) for both actor (noisy) and critic (clean),
* swaps xy-velocity tracking for an obstacle-gated variant and adds a lidar
  proximity penalty, so the policy ignores the velocity command while an
  obstacle blocks the commanded path and resumes tracking once clear.

The lidar plane clears stair risers out to ~0.9 m even at 23 cm steps, so the
avoidance terms do not fight the stair-climbing behavior inherited from StairsV2.

Real-robot note: deploying this policy requires the Go2 EDU's L1 lidar (or
equivalent) reduced to a 72-ray horizontal scan in addition to the elevation
map needed by the height scan.
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
    STAIRS_TERRAINS_CFG,
    RobotStairsV2EnvCfg,
    StairsObservationsCfg,
)

from . import obstacle_mdp

LIDAR_MAX_RANGE = 5.0

# StairsV2 terrain mix + tall box pillars (proportions are re-normalized by the
# generator, so the obstacle rows take ~1/6 of the columns)
OBSTACLE_TERRAINS_CFG = STAIRS_TERRAINS_CFG.copy()
OBSTACLE_TERRAINS_CFG.sub_terrains["obstacles"] = terrain_gen.MeshRepeatedBoxesTerrainCfg(
    proportion=0.25,
    # the cfg's default ("{DIR}.utils:make_box") doesn't resolve in this Isaac
    # Lab version; the generator looks up make_{object_type} by name
    object_type="box",
    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=4, height=1.5, size=(0.35, 0.35)
    ),
    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=12, height=1.5, size=(0.6, 0.6)
    ),
    # keep pillars tall at every difficulty so the elevated lidar plane always
    # sees them; curriculum scales count and footprint instead
    abs_height_noise=(0.0, 0.0),
    platform_width=1.5,
    platform_height=0.0,  # default -1 would spawn the robot on a 1.5 m pedestal
)


@configclass
class ObstacleSceneCfg(RobotSceneCfg):
    """Adds a horizontal 2D lidar to the stairs scene."""

    lidar_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        # ~0.7 m above ground when standing: clears stair risers <0.9 m ahead,
        # always intersects the 1.5 m pillars
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.4)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,  # 72 rays
        ),
        max_distance=2 * LIDAR_MAX_RANGE,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ObstacleObservationsCfg(StairsObservationsCfg):
    """Adds the lidar scan on top of the stairs observations."""

    @configclass
    class PolicyCfg(StairsObservationsCfg.PolicyCfg):
        # noisy scan: stand-in for a real 2D-reduced lidar
        lidar_scan = ObsTerm(
            func=obstacle_mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner"), "max_range": LIDAR_MAX_RANGE},
            noise=Unoise(n_min=-0.04, n_max=0.04),
            clip=(0.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(StairsObservationsCfg.CriticCfg):
        lidar_scan = ObsTerm(
            func=obstacle_mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner"), "max_range": LIDAR_MAX_RANGE},
            clip=(0.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class ObstacleRewardsCfg(RewardsCfg):
    """StairsV2 rewards with obstacle-gated tracking and a lidar proximity penalty."""

    track_lin_vel_xy = RewTerm(
        func=obstacle_mdp.track_lin_vel_xy_obstacle_gated,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": SceneEntityCfg("lidar_scanner"),
            "gate_distance": 1.0,
            "cone_half_angle": 1.05,  # ~60 deg
            "max_range": LIDAR_MAX_RANGE,
        },
    )
    obstacle_proximity = RewTerm(
        func=obstacle_mdp.lidar_obstacle_proximity,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("lidar_scanner"),
            "threshold": 0.8,  # below the ~0.9 m stair-riser intersection distance
            "max_range": LIDAR_MAX_RANGE,
        },
    )
    # StairsV2 drops undesired_contacts entirely (calf-on-step is legitimate);
    # re-introduce it for the head only, which should never touch anything
    head_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )


@configclass
class RobotObstacleEnvCfg(RobotStairsV2EnvCfg):
    """StairsV2 + obstacle avoidance in a single policy."""

    scene: ObstacleSceneCfg = ObstacleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObstacleObservationsCfg = ObstacleObservationsCfg()
    rewards: ObstacleRewardsCfg = ObstacleRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # the stairs post-init installs STAIRS_TERRAINS_CFG unconditionally,
        # so the obstacle mix (and its curriculum flag) must be set after it
        self.scene.terrain.terrain_generator = OBSTACLE_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt


@configclass
class RobotObstaclePlayEnvCfg(RobotObstacleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
