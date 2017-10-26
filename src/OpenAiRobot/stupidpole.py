import gym
env = gym.make('Pendulum-v0')
env.reset()
done = False
while not done:
    env.render()
    obs, rew, done,_ = env.step(env.action_space.sample()) # take a random action
    print("obs: {}, rew: {}, done: {}".format(obs,rew,done))