import gym


env = gym.make('MountainCarContinuous-v0')
env.reset()
tot_reward = 0
obs = [0,0]
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs[1] *= 1000
    obs, r, done, info = env.step([-2]) # take a random action
    tot_reward += r
    print(obs[1])
    a = env.observation_space
    e = env.action_space
    if done:
        break
