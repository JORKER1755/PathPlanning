import numpy as np
from .DQN import DQN

"""增强了行为策略的探索"""

class DQN2(DQN):
    def __init__(self,
                 model,
                 exp_buffer,
                 epsilon_linspace=(1., .1, 10),
                 gamma=0.9,
                 syn_freq=5,
                 **kwargs):
        super().__init__(model, exp_buffer, None, gamma, syn_freq)
        self.epsilons = np.linspace(*epsilon_linspace)
        self.n_epsilons = len(self.epsilons)

    def sample(self, state, task_percentage=0.):
        """decay epsilon: task_percentage∈[0,1)"""
        self.epsilon = self.epsilons[int(task_percentage * self.n_epsilons)]  # 0, 1, ..., n_epsilons-1
        return super().sample(state)
