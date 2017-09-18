import gym


class GymEnvironment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.obs = self.env.reset()
    def step(self, action):
        assert action == 0 or action == 1
        return self.env.step(action)
    def render(self):
        self.env.render()
    def set_max_steps(self, steps):
        self.env._max_episode_steps = steps
