import common
import options
from Run.data import DataPath, DataArgs
from Env.flight import Flight
import Algo
import Scene
from Run.exec import Train, Predict


__agent = None

def default_agent(env, algo_t=options.dqn, model_t=options.M_CNN):
    """简化使用难度，设计得不够好"""
    global __agent
    if __agent is None:
        algo = Algo.AlgoDispatch(algo_t, model_t)
        supervisor = algo(buffer_size=500)
        model = supervisor(env.d_states, env.action.n_actions, critic_lr=0.001)  #
        __agent = model(size_splits=env.d_states_detail, actor_n_layers=(20, 10), critic_n_layers=(20, 20, 20), n_filters=5,
                        state_n_layers=(20,))
    return __agent


def training(env, agent, paths, train_scenes, predict_scenes):
    print("\ntraining: start")
    train = Train(draw=False, episodes=args.episodes, max_n_step=args.max_n_step)
    predict = Predict(draw=common.use_windows, draw_rate=0.01, max_n_step=args.max_n_step, save=True)
    agent.build_model()
    for i in range(args.nth_rounds):  # 运行轮数
        # print('current round: ', i)
        train(env, agent, train_scenes, paths)
        agent.save_model(paths.model_save_file())
        predict(env, agent, predict_scenes, paths)
    print("training: end")

def prediction(env, agent, paths, predict_scenes):
    print("\nprediction: start")
    predict = Predict(draw=common.use_windows, draw_rate=args.draw_rate, max_n_step=args.max_n_step)
    """View only the last round of training results """
    agent.load_model(paths.model_load_file())
    predict(env, agent, predict_scenes, paths)

    """View training results for each round, or customize """
    # for nth_rounds_paths in paths:
    # for nth_rounds_paths in paths.rounds():
    #     agent.load_model(nth_rounds_paths.model_load_file())
    #     predict(env, agent, predict_scenes, nth_rounds_paths)
    print("prediction: end")

def run(args):
    if args.obstacle_type == options.O_circle:   # circle
        train_task = Scene.circle_train_task4x200
        test_task = Scene.circle_test_task4x100
    else:
        train_task = Scene.line_train_task4x200
        test_task = Scene.line_test_task4x100

    scene = Scene.ScenarioLoader()
    print("\ntraining scenes info: ")
    train_scenes = scene.load_scene(**train_task, percentage=args.percentage)
    print("\nprediction scenes info: ")
    predict_scenes = scene.load_scene(**test_task)

    env = Flight()
    agent = default_agent(env, args.algo_t, args.model_t)
    paths = DataPath(args)

    if args.train:
        training(env, agent, paths, train_scenes, predict_scenes)
    else:
        prediction(env, agent, paths, predict_scenes)


if __name__ == "__main__":
    # nth_times=k means the kth training/prediction
    # for training, -1 means to use the next version number; for prediction, it means the latest training
    args = DataArgs(train=False, nth_times=-1)
    args.max_n_step = 100   # Maximum number of steps per episode
    """for prediction"""
    args.draw_rate = 0.05  # Only draw the top xxx percent of all scenarios
    """for trainning"""
    args.episodes = 10  # the number of episodes per round, each round will perform model prediction and save
    args.nth_rounds = 3  # the number of rounds per training, total episodes per training = nth_rounds*episodes
    args.percentage = 0.25  # Only use the top xxx percent of all scenarios for training
    print(args.__dict__)

    run(args)
