"""
scene = task + obstacle
"""
from common import project_dir
import os
import numpy as np
from Env.flight import Scenairo


D_RANDOM, D_TYPICAL, D_OTHER = 'random', 'typical', 'other'         # data_source
T_TEST, T_EASY, T_HARD = np.arange(3)

circle_train_task4x200 = {'data_source': D_RANDOM, 'task_id': T_HARD, 'obs_id': 100}
circle_test_task4x100 = {'data_source': D_RANDOM, 'task_id': T_HARD, 'obs_id': 101}
line_train_task4x200 = {'data_source': D_RANDOM, 'task_id': T_HARD, 'obs_id': 200}
line_test_task4x100 = {'data_source': D_RANDOM, 'task_id': T_HARD, 'obs_id': 201}


class ScenarioLoader:
    circle_obs_name = 'circle_obs{}.npy'
    line_obs_name = 'line_obs{}.npy'

    task_map = {T_HARD: ([(50, 50), (50, 650), (650, 650), (650, 50)],      # init_pos
                         [45, -45, -135, 135],                              # init_dir
                         [(650, 650), (650, 50), (50, 50), (50, 650)],      # goal_pos
                         [45, -45, -135, 135]),                             # goal_dir
                T_EASY: ([[150, 150], [150, 550], [550, 550], [550, 150]],
                         [45, -45, -135, 135],
                         [[550, 550], [550, 150], [150, 150], [150, 550]],
                         [45, -45, -135, 135]),
                T_TEST: ([(50, 50)],
                         [45],
                         [(650, 650)],
                         [45])}

    def __init__(self):
        """延迟加载，提供更强的灵活性"""
        self.tsks, self.obss = None, None
        self.scene_dir = None
        self.n_scenarios = None
        self.scenarios = None

    def load_scene(self, obs_id, task_id=None, data_source=D_RANDOM, percentage=1.):
        self.scene_dir = project_dir('Scene', data_source, 'scene_files')
        independent = True
        if task_id is None:     # coupling任务要求obs_id和task_id必须相同，因此仅需提供obs_id即可
            task_id = obs_id
            independent = False

        circle_file = self.scene_dir.join(self.circle_obs_name.format(obs_id))
        if os.path.exists(circle_file):
            self.obss = [obs.astype(np.float) for obs in np.load(circle_file, allow_pickle=True)]
            build = self.independent_circle_scene if independent else self.coupling_circle_scene
        else:
            line_file = self.scene_dir.join(self.line_obs_name.format(obs_id))
            self.obss = [obs.astype(np.float) for obs in np.load(line_file, allow_pickle=True)]
            build = self.independent_line_scene if independent else self.coupling_line_scene
        if independent:  # 任务与障碍是组合关系(independent)还是一一对应(coupling)
            self.tsks = self.select_task(task_id)
        else:
            self.tsks = self.load_task(task_id)

        self.scenarios = build()      # 必须放在最后，保证依赖的数据均可使用
        self.n_scenarios = int(percentage * len(self.scenarios))
        self.scenarios = self.scenarios[:self.n_scenarios]
        print('obs_id: {}, task_id: {}, n_scenarios: {}, percentage: {}'.format(obs_id, task_id, self.n_scenarios, percentage))

    def load_task(self, task_id):
        init_p_file = self.scene_dir.join('start_pos{}.npy'.format(task_id))
        init_d_file = self.scene_dir.join('init_dir{}.npy'.format(task_id))
        goal_p_file = self.scene_dir.join('goal_pos{}.npy'.format(task_id))
        goal_d_file = self.scene_dir.join('goal_dir{}.npy'.format(task_id))
        return list(zip(np.load(init_p_file, allow_pickle=True).astype(np.float),
                        np.load(init_d_file, allow_pickle=True).astype(np.float),
                        np.load(goal_p_file, allow_pickle=True).astype(np.float),
                        np.load(goal_d_file, allow_pickle=True).astype(np.float)))

    def select_task(self, mode):
        return list(zip(*self.task_map[mode]))

    def coupling_circle_scene(self):
        print("coupling_circle: task_size: {}, obstacles_size: {}".format(len(self.tsks), len(self.obss)))
        return [Scenairo(*tsk, obs) for tsk, obs in zip(self.tsks, self.obss)]

    def coupling_line_scene(self):
        print("coupling_line: task_size: {}, obstacles_size: {}".format(len(self.tsks), len(self.obss)))
        return [Scenairo(*tsk, line_obstacles=obs) for tsk, obs in zip(self.tsks, self.obss)]

    def independent_circle_scene(self):
        print("independent_circle: task_size: {}, obstacles_size: {}".format(len(self.tsks), len(self.obss)))
        return [Scenairo(*tsk, obs) for tsk in self.tsks for obs in self.obss]

    def independent_line_scene(self):
        print("independent_line: task_size: {}, obstacles_size: {}".format(len(self.tsks), len(self.obss)))
        return [Scenairo(*tsk, line_obstacles=obs) for tsk in self.tsks for obs in self.obss]
