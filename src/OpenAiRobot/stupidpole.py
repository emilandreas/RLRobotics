import gym
import numpy as np
import ContinuousCartPole
env = gym.make('ContinuousCartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = np.array([0])
    env.step(action) # take a random action
    print(action)