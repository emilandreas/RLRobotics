import gym


class Game:
    def __init__(self):
        self.env = gym.make('Breakout-v0')
    def get