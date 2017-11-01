import gym
import numpy as np
import ContinuousCartPole
env = gym.make('ContinuousCartPole-v0')
env.reset()
done = False
while not done:
    env.render()
    action = np.array([0])
    env.step(action)
    print(action)
