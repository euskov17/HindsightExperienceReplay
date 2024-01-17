import numpy as np


class BitFlipEnv:
    def __init__(self, num_bits=10):
        self.num_bits = num_bits
        self.observation = self.random_state()
        self.goal = self.random_state()

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

    def step(self, action):
        assert 0 <= action < self.num_bits

        old_obs = self.observation 
        self.observation[action] = 1 - self.observation[action]
        
        # print((old_obs != self.observation))
        # print(self.observation)
        done = np.allclose(self.observation, self.goal)
        # if done:
        #     print("Success")
        # else:
        #     dist = (self.observation != self.goal).sum()
        #     print(f"Distance = {dist}")
        reward = done - 1

        next_state = self.get_state()
        return next_state, reward, done, {}, {}

        
