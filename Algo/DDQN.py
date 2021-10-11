import numpy as np
from .RL import ValueBasedRL


class DDQN(ValueBasedRL):
    def __init__(self,
                 model,
                 exp_buffer,
                 epsilon=0.1,
                 gamma=0.9,
                 syn_freq=10,
                 **kwargs):
        super().__init__(model, exp_buffer, epsilon, syn_freq)
        self.gamma = gamma

    def learn(self, batch_size=32):
        self.synchronize_weights()
        s, a, r, s_, d = self.exp_buffer.sample(batch_size)
        target_value = self.model.value(s)      # ndarray
        a_ = np.argmax(self.model.value(s_), axis=1)
        next_state_value = self.model.target_value(s_)[np.arange(batch_size), a_]   # 或者使用tf.gather_nd()
        target = r + np.logical_not(d) * next_state_value * self.gamma
        target_value[np.arange(batch_size), a] = target
        loss, metric = self.model.train(s, target_value)
        # print('current loss: {0:10f}, metric: {1:10f}'.format(loss, metric))
        return loss, metric     # critic_loss, actor_loss
