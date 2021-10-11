import tensorflow as tf
import pickle


class ActionValueSupervisor:
    model_names = ['ActionModel', 'ValueModel']

    def __init__(self,
                 n_actions,
                 ActionModel,
                 action_special_info,
                 ValueModel,
                 value_special_info,
                 actor_lr=0.001,
                 critic_lr=0.01):
        self.n_actions = n_actions  # 动作个数
        self.ActionModel = ActionModel
        self.action_special_info = action_special_info
        self.ValueModel = ValueModel
        self.value_special_info = value_special_info
        self.critic_lr, self.actor_lr = critic_lr, actor_lr
        self.build = False

    def action(self, state):
        return self.sess.run(self.Actor.out, feed_dict=self.Actor(state))

    def value(self, state):
        return self.sess.run(self.Critic.out, feed_dict=self.Critic(state))  # 两者没有共享placeholder，效率有所降低

    def train_critic(self, state, value_target):
        td_error, critic_loss, _ = self.sess.run([self.td_error, self.critic_loss, self.critic_train],
                                                 feed_dict={**self.Critic(state), self.value_target: value_target})
        return critic_loss, td_error

    def train_actor(self, state, action, decayed_td_error):
        actor_loss, _ = self.sess.run([self.actor_loss, self.actor_train],
                                      feed_dict={**self.Actor(state), self._action: action,
                                                 self.td_error: decayed_td_error})
        return actor_loss, -actor_loss

    def _build_model(self):
        self.build = True
        self.Actor = self.ActionModel(self.n_actions, self.action_special_info, eval_model=True)
        self.Critic = self.ValueModel(1, self.value_special_info, eval_model=True)  # 状态价值而不是动作价值
        self.value_target = tf.placeholder(tf.float64, (None, 1), name='value_target')
        self.td_error = self.value_target - self.Critic.out
        self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
        self.critic_train = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, var_list=self.Critic.vars)
        self._action = tf.placeholder(tf.int32, (), name='action')
        self.actor_loss = -tf.reduce_mean(tf.log(self.Actor.out[0, self._action]) * self.td_error)  # decayed_td_error
        self.actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, var_list=self.Actor.vars)
        # Target Model
        self.TargetActor = self.ActionModel(self.n_actions, self.action_special_info, eval_model=False)
        self.TargetCritic = self.ValueModel(1, self.value_special_info, eval_model=False)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  #

    def build_model(self):
        self._build_model()

    def load_model(self, path):
        if not self.build:
            self._build_model()
        with open(path, 'rb') as fp:
            c_weights, a_weights = pickle.load(fp)
        for var, weight in zip(self.Critic.vars, c_weights):
            var.load(weight, self.sess)
        for var, weight in zip(self.Actor.vars, a_weights):
            var.load(weight, self.sess)

    def save_model(self, path):
        c_weights = self.sess.run(self.Critic.vars)
        a_weights = self.sess.run(self.Actor.vars)
        with open(path, 'wb') as fp:
            pickle.dump((c_weights, a_weights), fp)
