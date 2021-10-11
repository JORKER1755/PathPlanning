import numpy as np
from .RL import ValueBasedRL


class DQN(ValueBasedRL):
    def __init__(self,
                 model,
                 exp_buffer,
                 epsilon=0.1,
                 gamma=0.9,
                 syn_freq=5,
                 **kwargs):
        super().__init__(model, exp_buffer, epsilon, syn_freq)
        self.gamma = gamma

    def learn(self, batch_size=32):
        self.synchronize_weights()
        s, a, r, s_, d = self.exp_buffer.sample(batch_size)
        target_value = self.model.value(s)      # ndarray
        next_state_value = np.max(self.model.target_value(s_), axis=1)
        target = r + np.logical_not(d)*next_state_value*self.gamma
        target_value[np.arange(batch_size), a] = target
        loss, metric = self.model.train(s, target_value)
        # print(loss, type(loss))
        return loss, metric     # critic_loss, actor_loss
