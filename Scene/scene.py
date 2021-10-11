import numpy as np
from Scene.draw_scene import DrawScene
import math


def yxzip(xs, y_func):  # {(x,y)|y=y(x)}
    for x in xs:
        yield x, y_func(x)


def xyzip(x_func, ys):  # {(x,y)|x=x(y)}
    for y in ys:
        yield x_func(y), y


def xy_zip(_xs, _ys):
    """将y=y(x)和x=x(y)统一地模拟zip: {(x, y)}"""
    x_callable, y_callable = callable(_xs), callable(_ys)
    if x_callable or y_callable:
        _zip = xyzip if x_callable else yxzip
    else:
        _zip = zip
    return _zip(_xs, _ys)


def connect_points(points, axis=1):
    """含首尾，默认批量进行"""
    line_ob = []
    for i in range(len(points)):
        line_ob.append(np.concatenate([points[i - 1], points[i]], axis=axis))
    return np.array(line_ob).reshape((-1, 4))


def rotate(theta=45, clockwise=True):
    """必须是ndarray，shape：(2,) or (n, 2)"""
    theta = math.radians(-theta if clockwise else theta)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    rotate_matrix = np.array([[c_theta, s_theta], [-s_theta, c_theta]])

    def _rotate(pos): return pos @ rotate_matrix  # 支持shape=(2,)

    return _rotate


def rotate_circle(theta=45, clockwise=True):
    """原址旋转，必须是ndarray，shape：(3,) or (n, 3)"""
    rot = rotate(theta, clockwise)

    def _rotate(circle): circle[..., :2] = rot(circle[..., :2])  # 支持shape=(3,)

    return _rotate


def circle_rotate_translate(_centers, _radius, _rotate, _offset):
    centers = _rotate(np.array(_centers)) + _offset
    radiuss = np.full((len(centers), 1), _radius)
    return np.concatenate([centers, radiuss], axis=1)


class CircleObs:
    def __init__(self, n_obs_range, x_range, y_range, r_range):
        """[low, high]"""
        self.n_obs_range = self.enclosed(n_obs_range)  # [low, high]
        self.x_range = self.enclosed(x_range)
        self.y_range = self.enclosed(y_range)
        self.r_range = self.enclosed(r_range)  # 半径

    @staticmethod
    def enclosed(value_range):
        low, high = value_range
        return low, high + 1

    def random_generate(self, n_scene):
        obs_nums = np.random.randint(*self.n_obs_range, (n_scene,))  # 每个场景的障碍数量
        obs_batch = []
        for obs_num in obs_nums:
            xs = np.random.randint(*self.x_range, (obs_num, 1))
            ys = np.random.randint(*self.y_range, (obs_num, 1))
            radius = np.random.randint(*self.r_range, (obs_num, 1))
            obs_batch.append(np.concatenate((xs, ys, radius), axis=1))
        return np.array(obs_batch, dtype=np.object)


def simple_circle_obs(n_scene=1):
    cricle = CircleObs(n_obs_range=(2, 3), x_range=(250, 450), y_range=(250, 450), r_range=(30, 80))
    return cricle.random_generate(n_scene)


def complex_circle_obs(n_scene=1):
    cricle = CircleObs(n_obs_range=(6, 11), x_range=(150, 550), y_range=(50, 650), r_range=(30, 80))
    return cricle.random_generate(n_scene)


class LineObs:
    def __init__(self, n_obs_range, x_range, y_range, size_range):
        """rectangle: center+size"""
        self.n_obs_range = self.enclosed(n_obs_range)  # [low, high]
        self.x_range = self.enclosed(x_range)
        self.y_range = self.enclosed(y_range)
        self.size_range = self.enclosed(size_range)  # 半径

    @staticmethod
    def enclosed(value_range):
        low, high = value_range
        return low, high + 1

    def random_generate(self, n_scene):
        obs_nums = np.random.randint(*self.n_obs_range, (n_scene,))  # 每个场景的障碍数量
        obs_batch = []
        for obs_num in obs_nums:
            xs = np.random.randint(*self.x_range, (obs_num, 1))
            ys = np.random.randint(*self.y_range, (obs_num, 1))
            center = np.concatenate([xs, ys], axis=1)
            size = np.random.randint(*self.size_range, (obs_num, 2))
            half_size = size // 2  # 主对角，朝向右上角
            anti_half_size = half_size * np.array([1, -1])  # 反对角，朝向右下角
            rectangles = [center - half_size, center + anti_half_size, center + half_size,
                          center - anti_half_size]  # 逆时针
            obs_batch.append(connect_points(rectangles))
        return np.array(obs_batch, dtype=np.object)


def simple_line_obs(n_scene=1):
    line = LineObs(n_obs_range=(2, 3), x_range=(250, 450), y_range=(250, 450), size_range=(50, 160))
    return line.random_generate(n_scene)


def complex_line_obs(n_scene=1):
    line = LineObs(n_obs_range=(6, 7), x_range=(150, 550), y_range=(50, 650), size_range=(50, 160))
    return line.random_generate(n_scene)


tasks = {'easy': ([(50, 50), (50, 650), (650, 650), (650, 50)],
                  [(650, 650), (650, 50), (50, 50), (50, 650)],
                  [45, -45, -135, 135]),
         'hard': ([[150, 150], [150, 550], [550, 550], [550, 150]],
                  [[550, 550], [550, 150], [150, 150], [150, 550]],
                  [45, -45, -135, 135]),
         'test': ([(50, 50)],
                  [(650, 650)],
                  [45])}


def final_circle_scene():
    draw = DrawScene()
    draw.set_task([150, 150], [550, 550])
    obs = complex_circle_obs(100)
    [draw.draw_obstacle(obs, delay=0.8) for obs in obs[:30]]
    a = input('save: ')
    if a == 'y':
        np.save('random/scene_files/circle_obs101', obs)


def final_line_scene():
    draw = DrawScene()
    draw.set_task([150, 150], [550, 550])
    obs = complex_line_obs(100)
    [draw.draw_obstacle(line_obstacles=obs, delay=0.8) for obs in obs[:30]]
    a = input('save: ')
    if a == 'y':
        np.save('random/scene_files/line_obs201', obs)


def pick_circle_scene():
    obs = np.load('random/scene_files/circle_obs101.npy', allow_pickle=True)
    tsks = zip([(50, 50), (50, 650), (650, 650), (650, 50)],
                         [(650, 650), (650, 50), (50, 50), (50, 650)],
                         [45, -45, -135, 135])
    draw = DrawScene()
    for tsk in tsks:
        draw.set_task(*tsk)
        draw.draw_obstacle(obs[20], delay=0.8)
        a = input('continue: ')


if __name__ == '__main__':
    pick_circle_scene()

    # final_circle_scene()
    # final_line_scene()
    # r = rotate(clockwise=False)
    # print(r([1, 0]))
    # print(x)
    # x = xy_zip(range(4), lambda x:2*x)
    # x = xy_zip(lambda x:2*x, [2,3,4])
    # x = xy_zip([4,3,2], [2,3,4])
    # print(list(x))
    # _obstacles = BuildingObs().random_generate(10)
    # print(len(_obstacles))
    # [draw.draw_obstacle(obs, delay=0.8) for obs in _obstacles]
    # [draw.draw_obstacle(line_obstacles=obs, delay=0.8) for obs in _obstacles]
    # show_circle_ob(1)
    # show_line_ob(2)
    # compare()
    # pass
