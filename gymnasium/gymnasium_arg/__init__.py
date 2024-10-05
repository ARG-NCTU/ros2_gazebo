from gymnasium.envs.registration import make, register, registry, spec


register(
    id="blueboat-v1",
    entry_point="gymnasium_arg.envs:BlueBoat_V1",
    max_episode_steps=4096,
)