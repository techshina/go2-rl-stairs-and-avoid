"""Stairs + obstacle avoidance on a single EXTENDED fine grid (``Unitree-Go2-Velocity-ObstacleExt``).

In the height-map variant (``RobotHeightmapEnvCfg``) the coarse channel
perturbs the stair skill at runtime (15 cm summit 53% -> 89% when the coarse
input is clamped to a flat constant), so this task drops the coarse grid and extends the fine body grid instead
(single scanner, x -0.8..+2.0 m x y +-1.0 m at 0.1 m = 609 rays). See
``extgrid_mdp`` for the observation split (near block bit-compatible with the
stairs specialist x5 history + far extension without history) and the reward
terms.

Relative to the height-map variant (``RobotHeightmapEnvCfg``):

* sensors: ``coarse_scanner`` gone; ``height_scanner`` widened (same name, same
  z drop, offset +0.6 m so the pattern covers x in [-0.8, +2.0]),
* actor: 45x5 + near 187x5 + far 422 = 1,582 (was 2,525); critic 60 + 609 = 669 (was 520),
* rewards: same gate (1.0 m) / proximity (0.8 m) / head_contacts, classifier on
  the extended grid,
* terrain, terminations, curriculum, network: unchanged.

Warm start: the actor's first 1,160 input columns are laid out exactly as the
stairs specialist's, so specialist weights copy in bit-exactly with the far
block zero-initialized (src/train/ext_warmstart_surgery.py).
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    RobotSceneCfg,
)
from tools.stairs.velocity_stairs_env_cfg import RobotStairsV2EnvCfg

from . import extgrid_mdp
from .velocity_obstacle_env_cfg import OBSTACLE_TERRAINS_CFG

H_ABS = 0.35
H_STEP = 0.35


@configclass
class ExtSceneCfg(RobotSceneCfg):
    """Widens the body height scanner to x [-0.8, +2.0] x y [-1.0, +1.0] (0.1 m, 609 rays)."""

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(extgrid_mdp.EXT_X_OFFSET, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=extgrid_mdp.EXT_RES, size=list(extgrid_mdp.EXT_SIZE), ordering="xy"
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ExtObservationsCfg(ObservationsCfg):
    """Proprio + near block (specialist-compatible) + far extension."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # the stock 17x11 grid in the stock cell order: values, noise and clip match
        # the specialist's height_scan term, so its weights transfer bit-exactly
        height_scan_near = ObsTerm(
            func=extgrid_mdp.ext_height_scan_near,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # forward/side extension; wider clip because terrain 2 m ahead spans more height
        height_scan_far = ObsTerm(
            func=extgrid_mdp.ext_height_scan_far,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-2.0, 2.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        # privileged: the full extended grid, clean, single frame
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-2.0, 5.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class ExtRewardsCfg(RewardsCfg):
    """StairsV2 rewards with extended-grid gated tracking and proximity penalty."""

    track_lin_vel_xy = RewTerm(
        func=extgrid_mdp.track_lin_vel_xy_ext_gated,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "gate_distance": 1.0,
            "cone_half_angle": 1.05,  # ~60 deg
            "h_abs": H_ABS,
            "h_step": H_STEP,
        },
    )
    obstacle_proximity = RewTerm(
        func=extgrid_mdp.ext_obstacle_proximity,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
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
class RobotObstacleExtEnvCfg(RobotStairsV2EnvCfg):
    """Single extended fine grid, specialist-compatible near block."""

    scene: ExtSceneCfg = ExtSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ExtObservationsCfg = ExtObservationsCfg()
    rewards: ExtRewardsCfg = ExtRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = OBSTACLE_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        # per-term history: proprio and the near block keep the specialist's 5-frame
        # history; the far extension is a single frame (it changes slowly and its
        # history would only multiply noisy inputs). The StairsV2 post-init set a
        # group-level history of 5, which would override per-term values.
        self.observations.policy.history_length = None
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
            "height_scan_near",
        ):
            getattr(self.observations.policy, name).history_length = 5
        self.observations.policy.height_scan_far.history_length = 0


@configclass
class RobotObstacleExtPlayEnvCfg(RobotObstacleExtEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
