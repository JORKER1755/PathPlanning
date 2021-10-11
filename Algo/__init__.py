"""

"""
import Supervisor as sp


dqn, dqn2, ddqn, ddpg, ac = 'DQN', 'DQN2', 'DDQN', 'DDPG', 'AC'   # algorithm name
M_CNN, M_CNN2, M_CNN3, M_MLP, M_SHARED_CNN, M_SHARED_MLP = 'cnn', 'cnn2', 'cnn3', 'mlp', 'shared_cnn', 'shared_mlp'   # model_id


class AlgoDispatch:
    """算法分派
    首先选取algorithm/supervisor/model的类型，其次配置algorithm所需参数，然后配置supervisor所需参数，最后配置model所需参数
    信息提供的顺序：basic info => algo info => supervisor info => model info
    """
    import_algo_code = 'from .{0} import {0}'
    supervisor_map = {dqn: sp.S_VALUE, dqn2: sp.S_VALUE, ddqn: sp.S_VALUE, ddpg: sp.S_ACTOR_CRITIC, ac: sp.S_ACTION_VALUE}  # 配置算法关联的supervisor

    def __init__(self, algo_name, model_name):
        """basic info"""
        self.algo_name = algo_name
        self.model_name = model_name    # 用于作为保存数据的路径
        self.algo = self.import_algo()
        self.sv_dispatch = sp.SupervisorDispatch(self.supervisor_map[algo_name], model_name, self.build_algo)

    def __call__(self, **algo_kwargs):    # 算法参数，不包括model和exp_buffer
        """algo info"""
        self.algo_kwargs = algo_kwargs
        return self.sv_dispatch

    def build_algo(self, supervisor):
        """callback for SupervisorDispatch"""
        if self.algo.off_policy:
            from .util import ReplayMemory
            buffer = ReplayMemory(buffer_size=self.algo_kwargs['buffer_size'], action_dt=self.algo.action_type)
            return self.algo(supervisor, buffer, **self.algo_kwargs)
        else:
            return self.algo(supervisor, **self.algo_kwargs)

    def import_algo(self):
        exec(self.import_algo_code.format(self.algo_name))
        return locals()[self.algo_name]     # 方法内部不能使用globals()
