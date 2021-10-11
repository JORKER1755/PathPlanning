import common
"""
必须在工程根目录下，必须import common以将工程根目录添加到环境变量PATH中
"""
import argparse
from Run.run import run_baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('is_train', type=bool, help="train or predict")   #
    parser.add_argument('obstacle_type', type=str, help="'C':circle; 'L': line")   #
    parser.add_argument('episodes', type=int)       # 一次训练多少回合
    parser.add_argument('times', type=int)          # 多少次
    parser.add_argument('rounds', type=int)          # 多少轮
    parser.add_argument('percentage', type=float, help="0.25, 0.5, 0.75, 1.")   # # 使用训练场景的前百分之几
    parser.add_argument('max_n_step', type=int, help="100")   #

    run_baseline(parser.parse_args())
