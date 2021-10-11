import tensorflow as tf
import pickle


class ActorCriticSupervisor:
    model_names = ['ActorModel', 'CriticModel']

    def __init__(self,
                 ActorModel,
                 actor_special_info,
                 CriticModel,
                 critic_special_info,
                 actor_lr=0.001,
                 critic_lr=0.01):
        """区分Actor和Critic的placeholder，尽管两者形式相同但没有被共享"""
        self.ActorModel = ActorModel
        self.actor_special_info = actor_special_info
        self.CriticModel = CriticModel
        self.critic_special_info = critic_special_info
        self.critic_lr, self.actor_lr = critic_lr, actor_lr
        self.build = False

    def action(self, state):
        return self.sess.run(self.Actor.out, feed_dict=self.Actor(state))

    def target_action(self, state):
        return self.sess.run(self.TargetActor.out, feed_dict=self.TargetActor(state))

    def value(self, state):
        return self.sess.run(self.Critic.out, feed_dict={**self.Actor(state), **self.Critic(state)})    # 两者没有共享placeholder，效率有所降低

    def target_value(self, state):
        return self.sess.run(self.TargetCritic.out, feed_dict={**self.TargetActor(state), **self.TargetCritic(state)})

    def train_critic(self, state, action, value_target):
        critic_loss, _ = self.sess.run([self.critic_loss, self.critic_train], feed_dict={**self.Critic(state), self.Actor.out: action,
                                                    self.value_target: value_target})
        return critic_loss, 0.

    def train_actor(self, state):
        actor_loss, _ = self.sess.run([self.actor_loss, self.actor_train], feed_dict={**self.Actor(state), **self.Critic(state)})
        return actor_loss, 0.

    def build_model(self):
        self._build_model()
        self.synchronize_weights()

    def _build_model(self):
        self.build = True
        """critic的action如何处理"""
        self.Actor = self.ActorModel(self.actor_special_info, eval_model=True)
        self.Critic = self.CriticModel(self.Actor.out, self.critic_special_info, eval_model=True)
        self.actor_loss = -tf.reduce_mean(self.Critic.out)
        self.actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, var_list=self.Actor.vars)
        self.value_target = tf.placeholder(tf.float64, (None, 1), name='value_target')
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.value_target, self.Critic.out))
        self.critic_train = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, var_list=self.Critic.vars)
        # Target Model
        self.TargetActor = self.ActorModel(self.actor_special_info, eval_model=False)
        self.TargetCritic = self.CriticModel(self.TargetActor.out, self.critic_special_info, eval_model=False)
        self.tau = tf.placeholder(tf.float64, (), name='tau')
        self.syn_critic_vars = [tf.assign(t_var, (1.-self.tau)*t_var+self.tau*e_var) for t_var, e_var in zip(self.TargetCritic.vars, self.Critic.vars)]
        self.syn_actor_vars = [tf.assign(t_var, (1.-self.tau)*t_var+self.tau*e_var) for t_var, e_var in zip(self.TargetActor.vars, self.Actor.vars)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # 模型构建完才初始化

    def load_model(self, path):
        if not self.build:
            self._build_model()
        with open(path, 'rb') as fp:
            c_weights, a_weights = pickle.load(fp)
        for var, weight in zip(self.Critic.vars, c_weights):
            var.load(weight, self.sess)
        for var, weight in zip(self.Actor.vars, a_weights):
            var.load(weight, self.sess)
        self.synchronize_weights()

    def save_model(self, path):
        c_weights = self.sess.run(self.Critic.vars)
        a_weights = self.sess.run(self.Actor.vars)
        with open(path, 'wb') as fp:
            pickle.dump((c_weights, a_weights), fp)

    def synchronize_weights(self, tau=1.):
        self.sess.run(self.syn_critic_vars, feed_dict={self.tau: tau})
        self.sess.run(self.syn_actor_vars, feed_dict={self.tau: tau})
