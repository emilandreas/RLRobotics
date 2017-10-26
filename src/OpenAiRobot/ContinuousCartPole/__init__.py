from gym.envs.registration import register

register(
    id='ContinuousCartPole-v0',
    entry_point='ContinuousCartPole.ContinuousCartPole:ContinuousCartPole',
)
