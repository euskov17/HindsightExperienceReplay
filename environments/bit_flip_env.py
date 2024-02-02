import numpy as np
from gymnasium.spaces import Discret, Box

class BitFlipActionSpace:
    def __init__(self, n_bits):
        self.n_bits = n_bits
    
    def sample(self):
        return np.random.randint(0, self.n_bits, size=())

class BitFlipEnv:
    def __init__(self, num_bits=10):
        self.num_bits = num_bits
        self.observation_space = Box(low=0, high=1, shape=(num_bits,), dtype=np.int8)
        self.action_space = Discret(num_bits)

        self.observation = self.observation_space.sample()
        self.goal = self.observation_space.sample()

    def get_state(self):
        return {
            'observation' : self.observation.copy(),
            'achieved_goal' : self.observation.copy(),
            'desired_goal' : self.goal.copy()
        }
        
    def reset(self):
        self.observation = self.observation_space.sample()
        self.goal = self.observation_space.sample()
        return (self.get_state(), {})
    
    def compute_reward(self, achived_goal, desired_goal, info):
        done = np.allclose(achived_goal, desired_goal)
        reward = 0 if done else -1
        return reward

    def step(self, action):
        assert 0 <= action < self.num_bits
        action = int(action)
        self.observation[action] ^= 1
        
        done = np.allclose(self.observation, self.goal)

        reward = 0 if done else -1

        next_state = self.get_state()
        return next_state, reward, done, {}, {'is_success' : done}

        
