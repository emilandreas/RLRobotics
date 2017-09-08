import gym
import numpy as np
from imageFunctions import cropImg
from imageFunctions import rgb2gray
from imageFunctions import displayGray

VISUALIZE = True
inputChannels = 4
inputHeight = 160
inputWidth = 160
env = gym.make('Breakout-v0')
env.reset()
done = False

observations = np.empty((inputChannels, inputHeight, inputWidth))
#Each action is made k (4) steps in a row.

while not done:
    env.render()
    obs, reward, done, info = make_action(env.action_space.sample(), env, 1) # take a random action
    gray = rgb2gray(obs)
    grayCropped = cropImg(gray, inputHeight, inputWidth)
    displayGray(grayCropped)

    #print(env.unwrapped.get_action_meanings())
    a = 2
