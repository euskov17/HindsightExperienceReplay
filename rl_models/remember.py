from abc import ABC
from strategy import Strategy, FutureStrategy



class BufferRecorder(ABC):
    @abstract_method
    def record(self, trajectory, desired_goal, achived_goals, buffer, compute_reward):
        pass


class SimpleRecorder(BufferRecorder):
    def record(self, trajectory, desired_goal, achived_goals, buffer, compute_reward):
        for idx, tr in enumerate(trajectory):
            obs, action, reward, next_obs, done, info = tr

            obs_g = torch.cat([obs, desired_goal], -1)
            next_obs_g = torch.cat([next_obs, desired_goal], -1)
            buffer.add(obs_g, action, reward, next_obs_g, done)


class HER(BufferRecorder):
    def __init__(self, strategy : Strategy = FutureStrategy()):
        self.strategy = strategy

    def record(self, trajectory, desired_goal, achived_goals, buffer, compute_reward):
        steps = len(trajectory)
        n = np.random.randint(0, steps)
        idxes = np.random.randint(steps, size=n)
        for idx in idxes:
            additional_goals = self.strategy.sample(idx, achived_goals)
            obs, action, reward, next_obs, done, info = trajectory[idx]
            
            # Adding episode data
            obs_g = torch.cat([obs, desired_goal], -1)
            next_obs_g = torch.cat([next_obs, desired_goal], -1)
            buffer.add(obs_g, action, reward, next_obs_g, done)

            achived_goal = achived_goals[idx]
            
            # Adding her data
            for current_goal in additional_goals:
                reward = torch.tensor(compute_reward(
                                      achived_goal, 
                                      current_goal, None))
                done = reward != -1.0
                stategoal = torch.cat([obs, current_goal], -1)
                nextgoal = torch.cat([next_obs, current_goal], -1)
                buffer.add(stategoal, action, reward, nextgoal, done)

