import gym
import numpy as np
import matplotlib.pyplot as plt

from imageFunctions import cropImg
from imageFunctions import rgb2gray
from imageFunctions import displayGray

env = gym.make('Breakout-v0')
env.reset()
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(len(obs))
    print(len(obs[0]))
    gray = rgb2gray(obs)
    grayCropped = cropImg(gray, 160, 160)
    displayGray(grayCropped)
    plt.show()

    print(env.unwrapped.get_action_meanings())
    a = 2