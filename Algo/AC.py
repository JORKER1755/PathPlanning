import numpy as np
from .RL import OnPolicyRL


class AC(OnPolicyRL):
    def __init__(self,
                 model,
                 decay_rate=0.95,
                 gamma=0.9,
                 tau=0.05,
                 sigma=0.1,
                 **kwargs):
        super(AC, self).__init__(model)
        self.model = model
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.current_decay = None

    def sample(self, state):
        action_probs = np.squeeze(self.model.action(np.expand_dims(state, axis=0)), axis=0)
        print('ac', action_probs)
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)  # 这里float32计算精度不够

    def predict(self, state):
        return np.argmax(self.model.action(np.expand_dims(state, axis=0))[0])

    def reset(self):
        self.current_decay = 1.

    def learn(self, *exp):
        s, a, r, s_, d = exp
        s, s_ = np.expand_dims(s, axis=0), np.expand_dims(s_, axis=0)
        target_value = r + (not d) * self.gamma * self.model.value(s_)
        critic_loss, td_error = self.model.train_critic(s, target_value)
        # print('critic loss: {0:10f}, metric: {1:10f}'.format(critic_loss, np.squeeze(td_error)))
        # print(td_error, type(td_error))
        actor_loss, actor_metric = self.model.train_actor(s, a, td_error*self.current_decay)
        # print('actor loss: {0:10f}, metric: {1:10f}'.format(actor_loss, actor_metric))
        self.current_decay *= self.decay_rate
        return critic_loss, actor_loss
