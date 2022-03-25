"""
Manage experimental data
"""

import common
from utils import Version
import options
import time


class DataArgs:
    """Manage all parameters required by DataPath and provide default options """
    def __init__(self, train=False, nth_times=-1,
                 env_t=options.flight, algo_t=options.dqn, model_t=options.M_CNN,
                 obstacle_type=options.O_circle):
        self.train = train  # training or prediction
        #
        self.env_t = env_t
        self.algo_t = algo_t
        self.model_t = model_t
        self.obstacle_type = obstacle_type
        self.nth_times = nth_times


class DataPath:
    """调用顺序：首先是__init__, 其次是model_load_file(如果需要), 然后是__call__, 最后是model_save_file(如果需要)"""
    train, train_predict, test_predict = 'train_', 'train_pred_', 'test_pred_'   # run_type

    def __init__(self, args):
        """root_dir: 存放某次运行产生的数据的目录"""
        exp_dir = common.project_dir("Experiment", "{}_{}_{}_{}".format(args.env_t, args.algo_t, args.model_t, args.obstacle_type))
        if args.nth_times == -1:
            nth_exp_v = Version(exp_dir)
            if args.train:
                args.nth_times = nth_exp_v.next_available_version      # 第n次实验
            else:
                args.nth_times = nth_exp_v.latest_used_version
        # setattr(args, 'nth_times', nth_times)  #
        self.record_args(exp_dir, args)
        self.exp_dir = exp_dir(str(args.nth_times))
        print("\nnth_times: ", args.nth_times)

        self.shared_dir = self.exp_dir('shared_data')
        self.nth_rounds_v = Version(self.shared_dir)      # 第n轮

    def __call__(self, run_type):
        """创建与当前运行类型相关的数据文件"""
        if run_type == self.train:
            nth_rounds = self.nth_rounds_v.next_available_version  # 获得下一版本
            self.nth_rounds_v.update()                        # 更新
        else:
            nth_rounds = self.nth_rounds_v.latest_used_version     # 跟train的版本相同
        print(run_type, "nth_rounds: ", nth_rounds)
        self.private_dir = self.exp_dir(run_type + str(nth_rounds))
        # self.image_dir = self.private_dir('image')
        # self.picked_image_dir = self.private_dir('picked_image')
        self.task_record_file = self.private_dir.join('task_record')
        self.record_file = self.private_dir.join('record')
        # self.loss_file = self.private_dir.join('loss')
        # self.reward_file = self.private_dir.join('reward')
        # self.outcome_file = self.private_dir.join('outcome')
        return self

    @staticmethod
    def record_args(where, args):
        """执行记录，方便查询对应的运行数据"""
        args_file = where.join('args.txt')
        setattr(args, 'date', time.ctime())
        with open(args_file, 'a') as fp:
            fp.write(str(args.__dict__) + '\n')

    def model_save_file(self):
        return self.shared_dir.join('model_' + str(self.nth_rounds_v.latest_used_version))

    def model_load_file(self, version=None):
        if version is None:
            version = self.nth_rounds_v.latest_used_version
        # print("model_load_file", version)
        return self.shared_dir.join('model_' + str(version))

    def __iter__(self):
        """遍历所有的rounds"""
        self.iter_stop = self.nth_rounds_v.latest_used_version
        self.nth_rounds_v.latest_used_version = self.nth_rounds_v.initial_version
        return self

    def __next__(self):
        if self.nth_rounds_v.latest_used_version < self.iter_stop:
            self.nth_rounds_v.latest_used_version += 1
            return self
        else:
            raise StopIteration

    def __len__(self):
        return self.nth_rounds_v.latest_used_version

    def rounds(self, start=1, step=1, stop=None):
        """类似于range，[start, stop)"""
        backup = self.nth_rounds_v.latest_used_version
        if stop is None:
            stop = self.nth_rounds_v.latest_used_version + 1
        for nth in range(start, stop, step):
            self.nth_rounds_v.latest_used_version = nth
            yield self
        self.nth_rounds_v.latest_used_version = backup
