import gym

env = gym.make('MountainCarContinuous-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, r, done, info = env.step([action]) # take a random action
    print(env.observation_space)
    a = env.observation_space
    e = env.action_space
    if done:
        break
