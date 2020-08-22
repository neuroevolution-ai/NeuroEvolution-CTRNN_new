from gym.envs.registration import register

register(
    id="ReacherMemory-v0",
    entry_point="gym_memory_environments.envs:ReacherMemoryEnv",
    max_episode_steps=90  # Arbitrarily chosen, can be overwritten in the ReacherMemoryEnv class
)

register(
    id="ReacherMemoryDynamic-v0",
    entry_point="gym_memory_environments.envs:ReacherMemoryEnvDynamic",
    max_episode_steps=90  # Arbitrarily chosen, can be overwritten in the ReacherMemoryEnv class
)
