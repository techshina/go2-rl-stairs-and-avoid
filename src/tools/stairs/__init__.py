import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity-Stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_stairs_env_cfg:RobotStairsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_stairs_env_cfg:RobotStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tools.agents.rsl_rl_ppo_cfg:AsymmetricPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-StairsV2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_stairs_env_cfg:RobotStairsV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_stairs_env_cfg:RobotStairsV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tools.agents.rsl_rl_ppo_cfg:AsymmetricPPORunnerCfg",
    },
)
