import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity-Obstacle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_obstacle_env_cfg:RobotObstacleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_obstacle_env_cfg:RobotObstaclePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tools.agents.rsl_rl_ppo_cfg:AsymmetricPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-ObstacleHM",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_heightmap_env_cfg:RobotHeightmapEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_heightmap_env_cfg:RobotHeightmapPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tools.agents.rsl_rl_ppo_cfg:AsymmetricPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-ObstacleHMDiverse",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_heightmap_env_cfg:RobotHeightmapDiverseEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_heightmap_env_cfg:RobotHeightmapDiversePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tools.agents.rsl_rl_ppo_cfg:AsymmetricPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-ObstacleExt",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_extgrid_env_cfg:RobotObstacleExtEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_extgrid_env_cfg:RobotObstacleExtPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tools.agents.rsl_rl_ppo_cfg:AsymmetricPPORunnerCfg",
    },
)
