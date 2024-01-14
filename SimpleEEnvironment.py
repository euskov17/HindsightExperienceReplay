import numpy as np
class BitFlipEnvironment:
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.state = np.random.randint(0, 2, n_bits)
        self.goal = np.random.randint(0, 2, n_bits)
        self.n_actions = n_bits

    def get_n_actions(self):
        return self.n_actions
    
    def make_action(self, action_idx):
        self.state[action_idx] ^= 1
        return np.allclose(self.state, self.goal)

    def get_state(self):
        return self.state
    
    def get_goal(self):
        return self.goal