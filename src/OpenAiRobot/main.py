import gym

from imageFunctions import cropImg
from imageFunctions import rgb2gray
from imageFunctions import displayGray

VISUALIZE = True
stepsize = 4
inputHeight = 160
inputWidth = 160
env = gym.make('Breakout-v0')
env.reset()



#Each action is made k (4) steps in a row.
def makeAction(action, env, k):
    for i in range(k-1):
        env.step(action)
        if VISUALIZE:
            env.render()
    return env.step(action)

for _ in range(1000):
    env.render()
    obs, reward, done, info = makeAction(env.action_space.sample(), env, stepsize) # take a random action
    gray = rgb2gray(obs)
    grayCropped = cropImg(gray, inputHeight, inputWidth)
    displayGray(grayCropped)

    #print(env.unwrapped.get_action_meanings())
    a = 2
