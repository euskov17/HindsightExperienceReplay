import numpy as np
from gymnasium.spaces import Box

class Simple2d:
    def __init__(self, map):
        self.map = map
        self.shape = map.shape
        self.observation_space = Box(np.array([0.0, map.shape[0]]),
                                     np.array([0.0, map.shape[1]]),
                                     dtype=np.float32)
        self.action_space = Box(np.array([-1.0, 1.0]),
                                np.array([-1.0, 1.0]),
                                dtype=np.float32)
        self.observation = self.observation_space.sample()
        self.goal = self.random_state()

    def random_state(self):
        state = self.observation_space.sample()
        while not (state > 0).all() or self.map[state.astype(int)]:
            state = self.observation_space.sample()
        return state

    def get_state(self):
        return {
            'observation' : self.observation,
            'achieved_goal' : self.observation,
            'desired_goal' : self.goal
        }
    

    def reset(self):
        self.observation = self.random_state()
        self.goal = self.random_state()
        return (self.get_state(), {})
    
    def compute_reward(self, achived_goal, desired_goal, info):
        done = np.allclose(achived_goal, desired_goal)
        reward = 0 if done else -1
        return reward

    def check_correctness(self, state):
        if (state < 0).any():
            return False
        
        return not self.map[state.astype(int)]


    def step(self, action):
        st = self.observation
        collision_trouble = 0.0
        for i in range(100):
            nw = st + action / 100
            nw = nw.clamp(min=0.0)
            if not self.check_correctness(nw):
                collision_trouble -= 10
                break
            st = nw
        self.observation = st

        done = np.allclose(self.observation, self.goal)

        reward = 0 if done else -1
        reward += collision_trouble
        
        next_state = self.get_state()
        return next_state, reward, done, {}, {'is_success' : done}        
