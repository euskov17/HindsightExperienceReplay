import numpy as np

class BitFlipActionSpace:
    def __init__(self, n_bits):
        self.n_bits = n_bits
    
    def sample(self):
        return np.random.randint(0, self.n_bits, size=())

class BitFlipEnv:
    def __init__(self, num_bits=10):
        self.num_bits = num_bits
        self.observation = self.random_state()
        self.goal = self.random_state()
        self.action_space = BitFlipActionSpace(num_bits)

    def random_state(self):
        return np.random.randint(0, 2, size=self.num_bits)

    def get_state(self):
        return {
            'observation' : self.observation,
            'achieved_goal' : self.observation,
            'desired_goal' : self.goal
        }
    
    
    @property
    def n_actions(self):
        return self.num_bits
    
    @property
    def state_dim(self):
        return self.num_bits
    
    @property
    def start(self):
        return self.observation
    
    @property
    def finish(self):
        return self.goal

    def reset(self):
        self.observation = self.random_state()
        self.goal = self.random_state()
        return (self.get_state(), {})
    
    def compute_reward(self, achived_goal, desired_goal, info):
        done = np.allclose(achived_goal, desired_goal)
        reward = 0 if done else -1
        return reward

    def step(self, action):
        assert 0 <= action < self.num_bits
        action = int(action)
        self.observation[action] ^= 1 # - self.observation[action]
        
        done = np.allclose(self.observation, self.goal)

        reward = 0 if done else -1

        next_state = self.get_state()
        return next_state, reward, done, {}, {}

        
