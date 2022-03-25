import numpy as np
import math
# import functools


class Distance:
    sqrt2 = math.sqrt(2)

    @staticmethod
    def euclidean(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    @staticmethod
    def manhattan(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    @staticmethod
    def diagonal(self, other):
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)
        return dx + dy + (self.sqrt2 - 2) * min(dx, dy)

    @staticmethod
    def chebyshev(self, other):
        return max(abs(self.x - other.x), abs(self.y - other.y))


class Geometry:
    @staticmethod
    def dist(point):    # shape=(2, )               支持非ndarray
        return np.hypot(*point)

    @staticmethod
    def dist_batch(points):  # shape=(n, 2) or (2, )     必须为ndarray
        return np.hypot(*points.T)

    @classmethod
    def dist_p2seg(cls, p, p1, p2):
        """The shortest distance from point to line segment """
        p2p1, pp1 = p2 - p1, p - p1
        d = np.dot(p2p1, pp1)
        if d <= 0.: return cls.dist(pp1)
        d2 = np.dot(p2p1, p2p1)
        if d >= d2: return cls.dist(p - p2)
        return cls.dist(pp1 - p2p1 * d / d2)

    @classmethod
    def dist_p2arc(cls, p, center, radius, start, extent):
        """The shortest distance from a point to an arc"""
        # 中间区域的点，在pp1之上，在pp2之下，p1为圆弧起点，p2为圆弧终点，逆时针

        assert radius > 0.
        # 端点
        p1 = center + radius*np.array([np.cos(start), np.sin(start)])
        end = start + extent
        p2 = center + radius*np.array([np.cos(end), np.sin(end)])
        if extent < 0.:     # 顺时针，调整为逆时针
            p1, p2 = p2, p1

        # print('p1: {}, p2: {}'.format(p1, p2))

        cp1 = p1 - center
        cp2 = p2 - center
        cp = p - center
        cross1 = np.cross(cp1, cp)
        cross2 = np.cross(cp, cp2)
        if cross1 > 0. and cross2 > 0.:     # 中间区域
            dis = abs(radius - cls.dist(cp))
        else:
            dis1 = cls.dist(p - p1)
            dis2 = cls.dist(p - p2)
            dis = min(dis1, dis2)
        # print(cross1, cross2)
        return dis


class Trajectory:
    """过度封装会导致无法批量运算/矩阵运算"""
    LINE, ARC = 0, 1

    def __init__(self):
        self.traj_type = None
        self.traj = None

    def line2(self, start_pos, end_pos):
        self.traj_type = self.LINE
        self.traj = start_pos, end_pos

    def line(self, start_pos, delta_pos):
        self.traj_type = self.LINE
        self.traj = start_pos, start_pos+delta_pos

    def arc(self, start_pos, start_dir, radius, delta_dir):
        """
        delta_dir: 弧以顺时针为正，即右转
        radius: 有向半径，逆时针为负
        """
        self.traj_type = self.ARC
        center = start_pos + radius * np.array([np.sin(start_dir), -np.cos(start_dir)])
        start = start_dir + math.pi / 2
        if radius < 0.: start -= math.pi
        extent = -delta_dir
        self.traj = center, abs(radius), start, extent

    def __call__(self):
        return self.traj
