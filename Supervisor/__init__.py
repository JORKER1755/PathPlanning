import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
assert str(tf.__version__).startswith('1.'), "only tensorflow v1 is supported"

# from .util import ModelInfo

S_VALUE, S_ACTION_VALUE, S_ACTOR_CRITIC = 0, 1, 2  # supervisor_id


class SupervisorDispatch:
    def __init__(self, supervisor_id, model_id, build_algo):
        """对supervisor和model进行分派，并回调build_algo"""
        supervisors = {S_VALUE: self.value_supervisor,
                       S_ACTION_VALUE: self.action_value_supervisor,
                       S_ACTOR_CRITIC: self.actor_critic_supervisor}
        self.model_id = model_id
        self.special_info_names_list = None
        self.build_supervisor = supervisors[supervisor_id]()  # 依赖于model_id
        self.build_algo = build_algo

    def __call__(self, d_states, n_actions, actor_lr=0.001, critic_lr=0.01,
                 **kwargs):  # (self.d_obstacle_states, self.d_goal_states)
        """supervisor info"""
        self.d_states = d_states
        self.n_actions = n_actions
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        return self._model_info

    def _model_info(self, **model_kwargs):  # 独立+共享
        """model info"""
        tf.reset_default_graph()        # 复原tf避免重名
        return self.build_algo(self.build_supervisor(model_kwargs))  # 回调

    def import_models(self, model_names):
        models = []
        for model_name in model_names:
            exec('from Model.{} import {}'.format(self.model_id, model_name))
            models.append(locals()[model_name])
        return models

    def actor_critic_supervisor(self):
        from .actor_critic import ActorCriticSupervisor
        ActorModel, CriticModel = self.import_models(ActorCriticSupervisor.model_names)

        def _supervisor(special_info):
            return ActorCriticSupervisor(ActorModel, special_info,
                                         CriticModel, special_info, self.actor_lr,
                                         self.critic_lr)

        return _supervisor

    def action_value_supervisor(self):
        from .action_value import ActionValueSupervisor
        ActionModel, ValueModel = self.import_models(ActionValueSupervisor.model_names)

        def _supervisor(special_info):
            return ActionValueSupervisor(self.n_actions, ActionModel,
                                         special_info,
                                         ValueModel, special_info, self.actor_lr,
                                         self.critic_lr)

        return _supervisor

    def value_supervisor(self):
        from .value import ValueSupervisor
        ValueModel, = self.import_models(ValueSupervisor.model_names)  # 单元素解包，逗号不能省

        def _supervisor(special_info):
            return ValueSupervisor(self.n_actions, ValueModel,
                                   special_info, self.critic_lr)

        return _supervisor
