import tensorflow as tf
import pickle


class ValueSupervisor:
    model_names = ['ValueModel']  # 依赖的模型名

    def __init__(self,
                 n_actions,
                 ValueModel,
                 value_special_info,
                 critic_lr=0.001):
        self.ValueModel = ValueModel
        self.n_actions = n_actions  # 动作个数
        self.value_special_info = value_special_info
        self.critic_lr = critic_lr
        self.build = False  # build多次会导致self.Value.vars膨胀，因为vars是由tf全局保存的

    def _build_model(self):
        self.build = True
        self.Value = self.ValueModel(self.n_actions, self.value_special_info, eval_model=True)
        self.value_target = tf.placeholder(tf.float64, (None, self.n_actions), name='value_target')
        self.loss = tf.reduce_mean(tf.squared_difference(self.value_target, self.Value.out))
        self.train_critic = tf.train.AdamOptimizer(self.critic_lr).minimize(self.loss, var_list=self.Value.vars)
        self.TargetValue = self.ValueModel(self.n_actions, self.value_special_info, eval_model=False)
        self.syn_vars = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.TargetValue.vars, self.Value.vars)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # 模型构建完才初始化

    def build_model(self):
        self._build_model()
        self.synchronize_weights()

    def load_model(self, path):
        if not self.build:
            self._build_model()
        with open(path, 'rb') as fp:
            weights = pickle.load(fp)
        for var, weight in zip(self.Value.vars, weights):
            var.load(weight, self.sess)
        self.synchronize_weights()

    def save_model(self, path):
        weights = self.sess.run(self.Value.vars)
        with open(path, 'wb') as fp:
            pickle.dump(weights, fp)

    def train(self, state, value_target):
        loss, _ = self.sess.run([self.loss, self.train_critic],
                                feed_dict={**self.Value(state), self.value_target: value_target})
        return loss, 0.

    def value(self, state):
        return self.sess.run(self.Value.out, feed_dict=self.Value(state))

    def target_value(self, state):
        return self.sess.run(self.TargetValue.out, feed_dict=self.TargetValue(state))

    def synchronize_weights(self):
        self.sess.run(self.syn_vars)
