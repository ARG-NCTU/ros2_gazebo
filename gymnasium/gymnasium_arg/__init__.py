from gymnasium.envs.registration import make, register, registry, spec


register(
    id="multi-blueboat-v1",
    entry_point="gymnasium_arg.envs:Multi_BlueBoat_V1",
    max_episode_steps=4096,
)

register(
    id="blueboat-v1",
    entry_point="gymnasium_arg.envs:BlueBoat_V1",
    max_episode_steps=4096,
)

register(
    id="blueboat-v2",
    entry_point="gymnasium_arg.envs:BlueBoat_V2",
    max_episode_steps=4096,
)
