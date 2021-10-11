import numpy as np
from .RL import OffPolicyRL
import math


class DDPG(OffPolicyRL):
    action_type = float     # 连续动作

    def __init__(self,
                 model,
                 exp_buffer,
                 gamma=0.9,
                 tau=0.01,
                 sigma_linspace=(1., .1, 10),
                 **kwargs):
        super(DDPG, self).__init__(model, exp_buffer)
        self.exp_buffer = exp_buffer
        self.model = model
        self.gamma = gamma
        self.tau = tau
        self.sigmas = np.linspace(*sigma_linspace)
        self.n_sigmas = len(self.sigmas)

    def sample(self, state, task_percentage=0.):
        """decay sigma: task_percentage∈[0,1)"""
        sigma = self.sigmas[int(task_percentage * self.n_sigmas)]  # 0, 1, ..., n_sigmas-1
        raw_action = self.predict(state)
        noise = math.copysign(np.random.normal(0., sigma), -raw_action)
        # print("raw_action: {:.7f}, sigma: {:.3f}, noise: {:.3f}".format(raw_action, sigma, noise))
        return np.clip(raw_action + noise, -1., 1.)

    def predict(self, state):
        return np.squeeze(self.model.action(np.expand_dims(state, axis=0)))

    def synchronize_weights(self):
        self.model.synchronize_weights(self.tau)

    def learn(self, batch_size=32):
        self.synchronize_weights()
        s, a, r, s_, d = self.exp_buffer.sample(batch_size)
        a, r, d = np.expand_dims(a, axis=-1), np.expand_dims(r, axis=-1), np.expand_dims(d, axis=-1)
        next_state_value = self.model.target_value(s_)
        target_value = r + np.logical_not(d) * next_state_value * self.gamma
        critic_loss, critic_metric = self.model.train_critic(s, a, target_value)
        actor_loss, actor_metric = self.model.train_actor(s)
        return critic_loss, actor_loss
