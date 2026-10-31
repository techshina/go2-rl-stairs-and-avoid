from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class AsymmetricPPORunnerCfg(BasePPORunnerCfg):
    """Asymmetric actor-critic: the critic consumes the privileged ``critic``
    obs group (clean height scan, base_lin_vel, joint_effort) while the actor
    stays on the deployable noisy ``policy`` group."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
