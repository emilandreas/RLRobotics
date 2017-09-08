import numpy as np

class Agent:
    def __init__(self, env, network):
        self.network = network

        #the state is given by the last n (4?) frames, and saved in memory
        self.memory = np.empty(self.network.get_input_shape())
        self.memory_size, _, _ = self.network.get_input_shape()
    def 
    def __update_memory(self, obs):
        for i in reversed(range(1, self.memory_size)):
            self.memory[i, :, :] = self.memory[i-1, :, :]
        self.memory[0, :, :] = obs
