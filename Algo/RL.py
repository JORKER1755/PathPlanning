import numpy as np


class OffPolicyRL:
    """forward the methods from model or exp_buffer"""
    off_policy = True
    action_type = int   # 离散动作

    def __init__(self,
                 model,
                 exp_buffer):
        self.exp_buffer = exp_buffer
        self.model = model

    def store_exp(self, *exp):
        self.exp_buffer.append(exp)

    def build_model(self):
        self.model.build_model()

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)

    def sample(self, state, **kwargs):
        raise NotImplementedError

    def learn(self, batch_size=32):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def synchronize_weights(self):
        raise NotImplementedError


class OnPolicyRL:
    """forward the methods from model"""
    off_policy = False
    action_type = int   # 离散动作

    def __init__(self,
                 model):
        self.model = model

    def build_model(self):
        self.model.build_model()

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)

    def sample(self, state):
        raise NotImplementedError

    def reset(self):
        pass

    def learn(self, *exp):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError


class ValueBasedRL(OffPolicyRL):
    def __init__(self,
                 model,
                 exp_buffer,
                 epsilon,
                 syn_freq):
        super().__init__(model, exp_buffer)
        self.epsilon = epsilon
        self.syn_freq = syn_freq
        self.learn_count = 0
        self.exp_buffer = exp_buffer
        self.model = model

        self.values = None      # 动作价值，模型测试时使用

    def sample(self, state, **kwargs):
        if np.random.uniform() > self.epsilon:  # greedy   0.1
            return self.predict(state)
        else:
            return np.random.randint(0, self.model.n_actions)

    def predict(self, state):
        # return np.argmax(self.model.value(np.expand_dims(state, axis=0))[0])
        self.values = np.squeeze(self.model.value(np.expand_dims(state, axis=0)), axis=0)
        return np.argmax(self.values)

    def synchronize_weights(self):
        if self.learn_count % self.syn_freq == 0:
            self.model.synchronize_weights()
        self.learn_count += 1

    def learn(self, batch_size=32):
        raise NotImplementedError
