"""Stairs variant of the Go2 velocity task.

Extends :class:`RobotEnvCfg` with:

* rough terrains including pyramid stairs (the sub-terrain set that ships
  commented-out in ``velocity_env_cfg.py``, with Unitree's original
  proportions and parameters), and
* a ``height_scan`` exteroceptive observation for both actor (noisy) and
  critic (clean), enabling blind-free stair traversal. Pair this env with an
  asymmetric PPO runner cfg (``obs_groups={"actor": ["policy"], "critic":
  ["critic"]}``) so the critic sees the privileged clean scan.

Real-robot note: deploying this policy requires an elevation-mapping pipeline
(e.g. LiDAR -> elevation map) that reproduces the height-scan grid; the
proprioceptive-only tasks remain the ones deployable with unitree_sdk2 alone.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ObservationsCfg, RobotEnvCfg

STAIRS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class StairsObservationsCfg(ObservationsCfg):
    """Adds the height scan on top of the proprioceptive observations."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # noisy scan: stand-in for a real elevation map (LiDAR + drift + gaps)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RobotStairsEnvCfg(RobotEnvCfg):
    """Velocity tracking over rough terrain incl. stairs, with height scan."""

    observations: StairsObservationsCfg = StairsObservationsCfg()

    def __post_init__(self):
        # swap the terrain before the base post-init so the terrain-level
        # curriculum flag is applied to the stairs generator
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        super().__post_init__()
        # follow env 0's robot with the viewport camera (used by play/--video)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.8, 1.8, 0.9)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class RobotStairsPlayEnvCfg(RobotStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotStairsV2EnvCfg(RobotStairsEnvCfg):
    """StairsV2: rewards retuned for stair climbing.

    With the stock rewards (``RobotStairsEnvCfg``) training plateaus at terrain
    level ~1.1 (stairs up to ~11cm): the unitree_rl_lab reward set is tuned for
    smooth flat-ground walking and systematically punishes stair-climbing
    behavior. This variant moves the
    five conflicting terms toward the Isaac Lab rough-task values (which
    reached ~15cm steps):

    * flat_orientation_l2 -2.5 -> 0.0  (allow pitched climbing posture)
    * undesired_contacts  -1.0 -> None (calf contact with step edges is OK)
    * air_time_variance   -1.0 -> -0.1 (allow irregular stepping)
    * joint_pos           -0.7 -> -0.35 (allow bigger leg swings)
    * action_rate         -0.1 -> -0.05 (allow faster recovery moves)

    Also enables a 5-frame observation history for the actor (temporal cues
    help blind-ish step detection under the noisy height scan).
    """

    def __post_init__(self):
        super().__post_init__()
        # rewards conflicting with stair climbing (IL-rough style)
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.air_time_variance.weight = -0.1
        self.rewards.joint_pos.weight = -0.35
        self.rewards.action_rate.weight = -0.05
        # actor observation history (critic keeps single-frame true state)
        self.observations.policy.history_length = 5


@configclass
class RobotStairsV2PlayEnvCfg(RobotStairsV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
