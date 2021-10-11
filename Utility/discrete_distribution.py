import numpy as np


def exponential_weighted_average1(n, weight):
    """
    指数加权平均: weight^n, (1-weight)*weight^(n-1), ..., (1-weight)*weight, 1-weight
    """

    distribution = np.zeros(n+1, dtype=np.float32)
    coef = 1.0 - weight
    prod = 1.0
    for i in range(n, 0, -1):
        distribution[i] = coef*prod
        prod *= weight
    distribution[0] = prod
    return distribution


def exponential_weighted_average2(n, weight):
    """
    指数加权平均: (1-weight)^n, weight*(1-weight)^(n-1), ..., weight*(1-weight), weight
    """
    return exponential_weighted_average1(n, 1.0-weight)


if __name__ == '__main__':
    # d = exponential_weighted_average2(20, 1)
    # print(d)

    pass
