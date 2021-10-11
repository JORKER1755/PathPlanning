import common
from Utility.util import Version

from Run.util import Record
import Algo
import Scene
import Experiment as Exp

import numpy as np
import time
import math


class Train:
    def __init__(self, draw=False, episodes=200, batch_size=64, max_n_step=50):
        self.max_n_step = max_n_step
        self.draw = draw
        self.episodes = episodes
        self.batch_size = batch_size

    def __call__(self, context):
        self.env = context.env
        self.scene = context.scene
        self.agent = context.agent
        self._train = self.train_off_policy if self.agent.off_policy else self.train_on_policy
        self.path = context.path(DataPath.train)

        total_task_size = self.episodes * self.scene.n_scenarios
        # self.task_r = Record(total_task_size, self.path.task_record_file, {
        #     'id': int, 'n_step': int, 'outcome': int, 'n_learn': int,
        #     'total_reward': float, 'average_reward': float,
        #     'total_actor_loss': float, 'average_actor_loss': float,
        #     'total_critic_loss': float, 'average_critic_loss': float})
        self.outcomes = np.zeros(self.env.n_outcomes, np.int)
        time_start = time.time()
        self.current_step = 0
        for episode in range(self.episodes):
            print('current episode: ', episode)
            for task_id, scenario in enumerate(self.scene.scenarios):
                self._train(scenario, (episode * self.scene.n_scenarios + task_id) / total_task_size)
        run_time = time.time() - time_start
        with open(self.path.record_file, 'a') as fp:
            s = "date: {}\n" \
                "time: {}\n" \
                "step: {}\n" \
                "n_scenes: {}\n" \
                "episode: {}\n" \
                "outcome: {}\n"
            s = s.format(time.ctime(), run_time, self.current_step, self.scene.n_scenarios, self.episodes, self.outcomes)
            fp.write(s)
            print(s)
        # self.task_r.save()
        print('train end\n\n\n')
        return self

    def train_off_policy(self, scenario, task_percentage):
        observation = self.env.reset(scenario)
        # self.task_r.init()  # __enter__
        for i in range(self.max_n_step):
            action = self.agent.sample(observation, task_percentage=task_percentage)
            # print("goal_dis: {}, goal_dir: {}".format(self.env.goal_dis, self.env.goal_dir))
            observation_, reward, done = self.env.step(action)
            # print("out: {}, reward: {}".format(out, reward))
            # self.task_r.total_reward += reward
            if self.draw:
                self.env.render(sleep_time=0.001, show_arrow=True, show_scope=False)
            self.agent.store_exp(observation, action, reward, observation_, done)
            if self.current_step > self.batch_size and self.current_step % 5 == 0:
                c_loss, a_loss = self.agent.learn(self.batch_size)
                # self.task_r.total_critic_loss += c_loss
                # self.task_r.total_actor_loss += a_loss
                # self.task_r.n_learn += 1
                # if self.current_step % 100 == 0:
                #     print('c_loss: {0:10f}, a_loss: {1:10f}'.format(c_loss, a_loss))
            observation = observation_
            self.current_step += 1
            if done:
                # self.task_r.n_step = i + 1
                break
        # else:
        #     self.task_r.n_step = self.max_n_step
        self.outcomes[self.env.result] += 1
        # self.task_r.outcome = self.env.result
        # self.task_r.average_reward = self.task_r.total_reward / self.task_r.n_step
        # self.task_r.average_actor_loss = 0. if self.task_r.n_learn == 0 else self.task_r.total_actor_loss / self.task_r.n_learn
        # self.task_r.average_critic_loss = 0. if self.task_r.n_learn == 0 else self.task_r.total_critic_loss / self.task_r.n_learn
        # self.task_r.inc()  # __exit__ 必须放在最后

    def train_on_policy(self, scenario, task_percentage):
        observation = self.env.reset(scenario)
        self.agent.reset()
        # self.task_r.init()  # __enter__
        for i in range(self.max_n_step):
            # print(observation)
            action = self.agent.sample(observation)
            observation_, reward, done = self.env.step(action)
            # self.task_r.total_reward += reward
            if self.draw:
                self.env.render(sleep_time=0.01, show_arrow=True, show_scope=True)
            c_loss, a_loss = self.agent.learn(observation, action, reward, observation_, done)
            # self.task_r.total_critic_loss += c_loss
            # self.task_r.total_actor_loss += a_loss
            # self.task_r.n_learn += 1
            observation = observation_
            self.current_step += 1
            if done:
                # self.task_r.n_step = i + 1
                break
        # else:
        #     self.task_r.n_step = self.max_n_step
        self.outcomes[self.env.result] += 1
        # self.task_r.outcome = self.env.result
        # self.task_r.average_reward = self.task_r.total_reward / self.task_r.n_step
        # self.task_r.average_actor_loss = 0. if self.task_r.n_learn == 0 else self.task_r.total_actor_loss / self.task_r.n_learn
        # self.task_r.average_critic_loss = 0. if self.task_r.n_learn == 0 else self.task_r.total_critic_loss / self.task_r.n_learn
        # self.task_r.inc()  # __exit__ 必须放在最后


class Predict:
    def __init__(self, draw=True, draw_rate=0.2, test_predict=True, pick=False, debug=False, max_n_step=50, save=False):
        self.save = save
        self.draw = draw
        self.draw_rate = draw_rate  # 只绘制前百分之xxx
        self.max_n_step = max_n_step
        self.pick = pick
        self.debug = debug
        self.test_predict = test_predict
        self._run = self._pick_task if pick else self.predict

    def __call__(self, context):
        self.scene = context.scene
        self.env = context.env
        self.agent = context.agent
        self.env = context.env
        self.path = context.path(DataPath.test_predict if self.test_predict else DataPath.train_predict)
        self._run()
        return self

    def predict(self):
        time_start = time.time()
        # 记录信息：步数，路程，角速率切换，切换次数，最大切换角速率，结果
        if self.save:
            task_r = Record(self.scene.n_scenarios, self.path.task_record_file, {
                'id': int, 'n_step': int, 'outcome': int,
                'total_reward': float, 'average_reward': float,
                'total_distance': float, 'total_time': float, 'linear_velocity': float,
                'total_change_yaw_rate': float, 'average_change_yaw_rate': float, 'max_change_yaw_rate': float,
                'total_yaw_rate': float, 'average_yaw_rate': float, 'max_yaw_rate': float,
                'total_turning_angle': float, 'average_turning_angle': float, 'max_turning_angle': float})
        outcomes = np.zeros(self.env.n_outcomes, np.int)
        for scenario in self.scene.scenarios:
            observation = self.env.reset(scenario)
            # task_r.init()  # __enter__
            for i in range(self.max_n_step):
                action = self.agent.predict(observation)
                observation_, reward, done = self.env.step(action)
                if self.save:
                    task_r.total_reward += reward
                    task_r.total_distance += self.env.delta_dis
                    task_r.total_time += self.env.delta_time
                    task_r.total_change_yaw_rate += abs(self.env.delta_yaw_rate)
                    if abs(self.env.delta_yaw_rate) > abs(task_r.max_change_yaw_rate):
                        task_r.max_change_yaw_rate = self.env.delta_yaw_rate
                    task_r.total_yaw_rate += abs(self.env.yaw_rate)
                    if abs(self.env.yaw_rate) > abs(task_r.max_yaw_rate):
                        task_r.max_yaw_rate = self.env.yaw_rate
                    task_r.total_turning_angle += abs(self.env.delta_dir)
                    if abs(self.env.delta_dir) > abs(task_r.max_turning_angle):
                        task_r.max_turning_angle = self.env.delta_dir
                if self.save:
                    if self.draw and task_r.count < self.draw_rate*self.scene.n_scenarios:
                        self.env.render(sleep_time=0.01, show_arrow=False, show_scope=True, show_pos=True)
                else:
                    self.env.render(sleep_time=0.01, show_arrow=False, show_scope=False, show_pos=True)
                observation = observation_
                if done:
                    if self.save:
                        task_r.n_step = i + 1
                    break
            else:
                if self.save:
                    task_r.n_step = self.max_n_step
            outcomes[self.env.result] += 1
            # task_r.total_turning_angle = math.degrees(task_r.total_turning_angle)
            # task_r.max_turning_angle = math.degrees(task_r.max_turning_angle)
            if self.save:
                task_r.average_turning_angle = task_r.total_turning_angle / task_r.n_step  # change_times
                task_r.average_change_yaw_rate = task_r.total_change_yaw_rate / task_r.n_step  # change_times
                task_r.average_yaw_rate = task_r.total_yaw_rate / task_r.n_step  # change_times
                task_r.average_reward = task_r.total_reward / task_r.n_step
                task_r.outcome = self.env.result
                task_r.linear_velocity = self.env.action.linear_velocity
                task_r.inc()  # __exit__ 必须放在最后

        run_time = time.time() - time_start
        if self.save:
            with open(self.path.record_file, 'a') as fp:    # 这里还应该将Record中的记录项保存起来，否则不方便读取保存的record
                s = "date: {}\n" \
                    "time: {}\n" \
                    "n_scenes: {}\n" \
                    "outcome: {}\n"
                s = s.format(time.ctime(), run_time, self.scene.n_scenarios, outcomes)
                fp.write(s)
                print(s)
            task_r.save()
            setattr(self, 'task_r', task_r)  # 用于数据后续处理
        print('predict end\n\n\n')

    def _pick_task(self):
        # while True:
        #     task_id = int(input('scene_id: '))
        for task_id in range(100):
            if task_id < 0 or task_id >= self.scene.n_scenarios:
                break
            scenario = self.scene.scenarios[task_id]
            observation = self.env.reset(scenario)
            for i in range(self.max_n_step):
                if self.debug:
                    print("step: {}".format(i))
                    print("obs_dis: {}".format(self.env.obs_distances))
                    print("goal_dis: {}, goal_dir: {}".format(self.env.goal_dis, self.env.goal_dir))
                action = self.agent.predict(observation)
                observation_, reward, done = self.env.step(action)
                if self.debug:
                    print("reward: {}, out: {}".format(reward, action))
                    input('continue')
                # if i % 5 == 0:
                self.env.render(sleep_time=0.01, show_arrow=False, show_scope=True, show_pos=True, circle_scope=False)
                # else:
                #     self.env.render(sleep_time=0.01, show_arrow=False, show_scope=False, show_pos=True)

                observation = observation_
                if done:
                    if self.env.result == self.env.success:
                        print('success')
                        self.env.render_trajectory(self.env.residual_traj, show_pos=False)
                    break
            # input('screenshot: start')
            # time.sleep(2)
            self.env.canvas.screenshot(self.path.picked_image_dir.join(str(task_id)))
            print('save screenshot: {}'.format(task_id))


class DataPath:
    """调用顺序：首先是__init__, 其次是model_load_file(如果需要), 然后是__call__, 最后是model_save_file(如果需要)"""
    train, train_predict, test_predict = 'train_', 'train_pred_', 'test_pred_'   # run_type

    def __init__(self, root_dir, run_type=None):
        """root_dir: 存放某次运行产生的数据的目录"""
        self.root_dir = root_dir
        self.shared_dir = self.root_dir('shared_data')
        self.v = Version(self.shared_dir)
        if run_type is not None: self.run_type = run_type

    def __call__(self, run_type):
        """创建与当前运行类型相关的数据文件"""
        v_str = self.v.version_plusone_str if run_type == self.train else self.v.version_str
        self.private_dir = self.root_dir(run_type + v_str)
        # self.image_dir = self.private_dir('image')
        self.picked_image_dir = self.private_dir('picked_image')
        self.task_record_file = self.private_dir.join('task_record')
        self.record_file = self.private_dir.join('record')
        # self.loss_file = self.private_dir.join('loss')
        # self.reward_file = self.private_dir.join('reward')
        # self.outcome_file = self.private_dir.join('outcome')
        return self

    @property
    def model_save_file(self):
        return self.shared_dir.join('model' + self.v.version_str_plusplus)

    @property
    def model_load_file(self):
        return self.shared_dir.join('model' + self.v.version_str)

    @property
    def latest_model_file(self):
        return self.shared_dir.join('model' + str(self.v.version-1))  # 加载最新版本

    def __iter__(self):
        self.iter_stop = self.v.version
        self.v.version = 1
        return self

    def __next__(self):
        if self.v.version <= self.iter_stop:
            self.__call__(self.run_type)
            self.v.version += 1
            return self
        else:
            self.v.version = self.iter_stop     # 复原
            raise StopIteration

    def __len__(self):
        """迭代过程中禁用此函数"""
        return self.v.version


class Context:
    """某次运行中的某轮运行(Train or Test)所依赖的所有资源，Context实例将沿着各轮运行构成的序列传递"""
    def __init__(self, agent, env, scene, path, eagerly=False):
        self.agent = agent  # 还未build
        self.env = env
        self.scene = scene
        self.path = path
        self.eagerly = eagerly

    def dispatch(self, call):
        return call() if self.eagerly else call

    """场景的上文控制：默认是复用场景"""
    def load_scene(self, *args, **kwargs):
        def _load_scene():
            self.scene.load_scene(*args, **kwargs)  # 延迟加载是必要的
        return self.dispatch(_load_scene)

    """模型的上文控制：默认是复用模型"""
    def build_model(self):
        return self.dispatch(self.agent.build_model)

    def load_model(self, model_load_name=None):
        """目前还不支持tf静态图模式，原因是优化器没有重建，而模型重建后，各层的name不一样了; 不支持静态图模式，存在全局数据"""
        def _load_model():
            load_file = self.path.model_load_file if model_load_name is None else self.path.shared_dir.join(model_load_name)
            self.agent.load_model(load_file)
        return self.dispatch(_load_model)

    """模型的下文控制：默认是不保存模型"""
    def save_model(self, model_save_name=None):
        def _save_model():
            save_file = self.path.model_save_file if model_save_name is None else self.path.shared_dir.join(model_save_name)
            self.agent.save_model(save_file)
        return self.dispatch(_save_model)


class RunUnit:
    """执行流程：首先是above_ctrl，其次是execute，最后是below_ctrl；操作对象都是context"""
    def __init__(self, execute, context, above_ctrl, below_ctrl):
        self.context = context
        self.execute = execute
        self.above_ctrl = above_ctrl
        self.below_ctrl = below_ctrl

    def __call__(self):
        # 处理上文，复用上文数据
        for ctrl in self.above_ctrl:
            ctrl()
        # 运行
        self.execute(self.context)
        # 处理下文，是否保存数据
        for ctrl in self.below_ctrl:
            ctrl()


class RunSequence:
    """管理某次运行中的各轮运行"""
    def __init__(self, context, init_ctrl=None):
        self.context = context
        self.sequence = [] if init_ctrl is None else init_ctrl

    def append(self, execute, above_ctrl=None, below_ctrl=None):
        self.sequence.append(RunUnit(execute, self.context,
                                     [] if above_ctrl is None else above_ctrl,
                                     [] if below_ctrl is None else below_ctrl))

    def __call__(self, mode):
        start, end = mode
        return [_run() for _run in self.sequence[start:end + 1]]


def run(args):
    print('run')
    from Env.flight import Flight
    env = Flight(max_detect_range=120., min_turning_radius=10., detect_angle_interval=5, safe_dis=0.,
                 use_border=True)
    """选择算法，获取算法对象和可选的Sp分派器，获得Sp和可选的Model分派器，最终生成agent"""
    algo = Algo.AlgoDispatch(args.algo_t, args.model_t)
    supervisor = algo(buffer_size=500, gamma=0.9)   # , tau=0.01
    model = supervisor(env.d_states, env.action.n_actions, critic_lr=0.001)     # 学习率比较关键
    agent = model(size_splits=env.d_states_detail, actor_n_layers=(20, 10), critic_n_layers=(20, 10), n_filters=5,
                  state_n_layers=(20, 10))

    exp_dir = Exp.experiment_path(args)
    v = Version(exp_dir)
    setattr(args, 'start_times', v.version)  # 记录中的起始运行次数
    Exp.record_args(args)
    root_dir = exp_dir(v.version_str_plusplus)
    # result_path = Exp.experiment_path(args)
    # root_dir = result_path(Version(result_path).version_str_plusplus)
    context = Context(agent, env, Scene.ScenarioLoader(), DataPath(root_dir))  # 一次运行的上下文

    draw = common.use_windows

    seq = RunSequence(context)

    train = Train(draw=False, episodes=args.episodes)
    seq.append(train, [context.build_model(), context.load_scene(**Scene.easy_task4x40)], [context.save_model()])
    test = Predict(draw=draw, draw_rate=0.1, test_predict=False)
    seq.append(test)
    test = Predict(draw=draw, draw_rate=0.15)
    seq.append(test, [context.load_scene(**Scene.hard_task4x100)])

    train = Train(draw=False, episodes=args.episodes)
    seq.append(train, [context.load_scene(**Scene.hard_task4x50)], [context.save_model()])
    test = Predict(draw=draw, draw_rate=0.1, test_predict=False)
    seq.append(test)
    test = Predict(draw=draw, draw_rate=0.15)
    seq.append(test, [context.load_scene(**Scene.hard_task4x100)])

    # M_TRAIN, M_TRAIN_TEST, M_TEST, M_RETRAIN, M_RETRAIN_RETEST, M_RETEST, M_ALL, M_DEBUG, M_PICK = \
    #     (0, 1), (0, 2), (2, 2), (3, 4), (3, 5), (5, 5), (0, 5), (6, 6), (7, 7)  # 基本任务的组合模式
    mode_map = {'train': (0, 1), 'train_pred': (0, 2), 'test': (2, 2), 'retrain': (3, 4), 'retrain_pred': (3, 5),
                'retest': (5, 5), 'all': (0, 5), 'pick': (6, 6), 'debug': (7, 7)}
    seq(mode_map[args.run_m])


def tst_cyclic(agent, env, args):
    exp_dir = Exp.experiment_path(args)
    times = args.times
    setattr(args, 'start_times', times)     # 记录中的起始运行次数
    Exp.record_args(args)
    v = Version(exp_dir)
    root_dir = exp_dir(v.latest_version_str)      # str(times)
    context = Context(agent, env, Scene.ScenarioLoader(), DataPath(root_dir), eagerly=True)  # 立即执行

    if args.obstacle_type == 'C':   # circle
        train_task = Scene.circle_train_task4x200
        test_task = Scene.circle_test_task4x100
    else:
        train_task = Scene.line_train_task4x200
        test_task = Scene.line_test_task4x100
    draw = common.use_windows
    # draw = False
    context.load_model('model0')
    # Predict(draw=draw, draw_rate=0.03, test_predict=False, max_n_step=args.max_n_step)(context)
    context.load_scene(**test_task)
    Predict(draw=draw, draw_rate=1., max_n_step=args.max_n_step, save=True)(context)


def run_cyclic(agent, env, args):
    exp_dir = Exp.experiment_path(args)
    v = Version(exp_dir)
    setattr(args, 'start_times', v.version)     # 记录中的起始运行次数
    Exp.record_args(args)
    root_dir = exp_dir(v.version_str_plusplus)
    context = Context(agent, env, Scene.ScenarioLoader(), DataPath(root_dir), eagerly=True)  # 立即执行
    context.build_model()

    if args.obstacle_type == 'C':   # circle
        train_task = Scene.circle_train_task4x200
        test_task = Scene.circle_test_task4x100
    else:
        train_task = Scene.line_train_task4x200
        test_task = Scene.line_test_task4x100
    draw = common.use_windows
    # draw = False
    for i in range(args.rounds):    # 运行轮数
        print('current round: ', i)
        context.load_scene(**train_task, percentage=args.percentage)       # Scene.hard_task4x50
        Train(draw=False, episodes=args.episodes, max_n_step=args.max_n_step)(context)
        context.save_model()
        # Predict(draw=draw, draw_rate=0.03, test_predict=False, max_n_step=args.max_n_step)(context)
        context.load_scene(**test_task)
        Predict(draw=draw, draw_rate=0.01, max_n_step=args.max_n_step, save=True)(context)


def run_baseline(args):
    args.env_t = 'baseline'
    args.algo_t = Algo.dqn
    args.model_t = Algo.M_CNN

    print('run_baseline')
    print(args.__dict__)

    from Env.flight import Flight, DiscreteAction
    env = Flight(DiscreteAction(45, 10., math.pi / 60), max_detect_range=120., detect_angle_interval=5, safe_dis=0.,
                 use_border=True)
    algo = Algo.AlgoDispatch(args.algo_t, args.model_t)
    supervisor = algo(buffer_size=500, gamma=0.9)   # , tau=0.01
    model = supervisor(env.d_states, env.action.n_actions, critic_lr=0.001)
    agent = model(size_splits=env.d_states_detail, actor_n_layers=(20, 10), critic_n_layers=(20, 20, 20), n_filters=5,
                  state_n_layers=(20,))

    if args.is_train:
        run_cyclic(agent, env, args)
    else:
        tst_cyclic(agent, env, args)


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.is_train = False
            self.obstacle_type = 'C'    # 'C':circle; 'L': line
            self.episodes = 10  # 10
            self.times = 1
            self.rounds = 1     # 4
            self.percentage = 0.25        # 使用训练场景的前百分之几
            self.max_n_step = 100
    args = Args()
    run_baseline(args)

