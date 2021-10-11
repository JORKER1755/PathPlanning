""" 实时路径规划环境
坐标系：x轴水平向右，y轴竖直向上，原点位于矩形区域的左下角
宽度：x轴
高度：y轴
角度：以x轴为基准，逆时针为正，类外部使用角度制，类内部使用弧度制
动作：右转为正，顺时针旋转
离散动作：[0, 1, 2, 3, 4]    必须为int类型
连续动作：[-1, +1]           必须为float类型
绘图坐标系：x轴水平向右，y轴竖直向下，原点位于矩阵区域的左上角
"""

import numpy as np
import math
from Geometry.geometry import Geometry, Trajectory
from Draw.draw import Canvas


class Kinematic:
    def __init__(self):
        """不保存目标状态"""
        # self.turning_angle_precision = 1 / max_turning_angle  # 归一化：max_turning_angle->1 => 1->1/max_turning_angle
        self.turning_angle_precision = 1 / 45  # 1/45 rad = 1.2732395447351628°

        # 状态量
        self.init_pos, self.init_dir = None, None
        self.goal_pos, self.goal_dir = None, None
        self.last_pos, self.last_dir = None, None
        self.current_pos, self.current_dir = None, None

        self.traj = Trajectory()

    def init(self, init_pos, init_dir, goal_pos, goal_dir):
        self.init_pos, self.init_dir = np.array(init_pos, dtype=np.float), math.radians(init_dir)
        self.goal_pos, self.goal_dir = np.array(goal_pos, dtype=np.float), math.radians(goal_dir)
        self.current_pos, self.current_dir = self.init_pos, self.init_dir

    def transition(self, delta_dir, delta_dis):
        """time_step; yaw_rate; speed"""
        if abs(delta_dir) < self.turning_angle_precision:   # 近似认为向前直线飞行
            delta_pos = np.array([delta_dis * np.cos(self.current_dir), delta_dis * np.sin(self.current_dir)])
            self.traj.line(self.current_pos, delta_pos)
        else:
            radius = delta_dis / delta_dir                  # 有向半径，此处不能取绝对值，利用半径的正负号来统一左转和右转
            delta_x = radius * (np.sin(delta_dir - self.current_dir) + np.sin(self.current_dir))  # -delta_dir抵消右转角度为负
            delta_y = radius * (np.cos(delta_dir - self.current_dir) - np.cos(self.current_dir))
            delta_pos = np.array([delta_x, delta_y])
            self.traj.arc(self.current_pos, self.current_dir, radius, delta_dir)

        self.last_pos = self.current_pos
        self.current_pos = self.current_pos + delta_pos     # 禁止任何原址操作，不允许写成：self.current_pos += delta_pos
        self.last_dir = self.current_dir
        self.current_dir -= delta_dir                       # python数据类型，此处为非原址操作


class Scenairo:
    region_size = np.array([700, 700])
    center_pos = region_size//2

    def __init__(self, init_pos, init_dir, goal_pos, goal_dir, circle_obstacles=None, line_obstacles=None):
        self.init_pos, self.init_dir = init_pos, init_dir
        self.goal_pos, self.goal_dir = goal_pos, goal_dir
        self.circle_obstacles, self.line_obstacles = circle_obstacles, line_obstacles

    def __str__(self):
        return "init_p: {}, goal_pos: {}, init_dir: {}".format(self.init_pos, self.goal_pos, self.init_dir)


class Action:
    """提供从动作到控制输入间的转换"""
    def __init__(self,
                 max_turning_angle,         # 转弯角度
                 min_turning_radius,        # 转弯半径
                 max_angular_velocity       # 最大转弯角速度
                 ):
        """不保存目标状态"""
        self.max_turning_angle = math.radians(max_turning_angle)
        self.max_angular_velocity = max_angular_velocity
        self.linear_velocity = self.max_angular_velocity * min_turning_radius  # math.pi/6
        self.basic_time_step = self.max_turning_angle / self.max_angular_velocity
        self.basic_distance_step = self.linear_velocity * self.basic_time_step  # 固定的路程步长

    def __call__(self, action):
        raise NotImplementedError


class DiscreteAction(Action):
    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity,  # 最大转弯角速度
                 n_granularity=4, n_actions=5,
                 ):
        super().__init__(max_turning_angle, min_turning_radius, max_angular_velocity)
        # 动作设计：已知动作个数和max_angle
        self.n_granularity = n_granularity
        self.max_time_step = self.basic_time_step * self.n_granularity
        self.max_distance_step = self.basic_distance_step * self.n_granularity
        assert n_actions % 2 == 1
        self.n_actions = n_actions
        self.actions = np.linspace(-self.max_turning_angle, self.max_turning_angle, self.n_actions)
        self.forward_action = self.n_actions // 2
        self.n_angle_actions = self.n_actions
        self.n_actions *= self.n_granularity  # 4个粒度等级

    def __call__(self, action):
        dis_step_times, angle_action = divmod(action, self.n_angle_actions)
        self.delta_dis = self.basic_distance_step * (dis_step_times + 1)
        self.delta_dir = self.actions[angle_action]


class ContinueAction(Action):
    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity,  # 最大转弯角速度
                 n_granularity=4
                 ):
        super().__init__(max_turning_angle, min_turning_radius, max_angular_velocity)
        # 动作设计：已知动作个数和max_angle
        self.n_granularity = n_granularity
        self.max_time_step = self.basic_time_step * self.n_granularity
        self.max_distance_step = self.basic_distance_step * self.n_granularity

    def __call__(self, action):
        dis_step_times, angle_action = action
        self.delta_dis = self.basic_distance_step * (
                    dis_step_times * (self.n_granularity - 1) + self.n_granularity + 1) / 2  # [-1, 1] => [1, n]
        self.delta_dir = angle_action * self.max_turning_angle


class BLPAction(Action):
    def __init__(self,
                 max_turning_angle,  # 转弯角度
                 min_turning_radius,  # 转弯半径
                 max_angular_velocity,  # 最大转弯角速度
                 ):
        super().__init__(max_turning_angle, min_turning_radius, max_angular_velocity)

    def __call__(self, action):
        delta_dir, delta_t = action
        self.delta_dir = delta_dir
        self.delta_dis = self.linear_velocity * delta_t


class Flight(Kinematic):
    """考虑运动学模型，位置+方向，定时间步长，定速，最大转弯角速率、最小转弯半径"""
    timeout, failure, border, success, n_outcomes = 0, 1, 2, 3, 4

    def __init__(self,
                 action=DiscreteAction(45, 10., math.pi/60, 4, 5),
                 max_detect_range=120.,  # 探测范围
                 detect_angle_interval=5,  # 距离传感器间的角度间隔
                 max_detect_angle=90,  # 最大探测角度
                 safe_dis=0.,  # 安全距离
                 canvas_size=None,  # 画布大小
                 add_noise=False,  # 是否对距离传感器施加噪声
                 use_border=False):
        super().__init__()
        self.region_width, self.region_height = Scenairo.region_size
        self.canvas = Canvas(Scenairo.region_size if canvas_size is None else canvas_size, Scenairo.region_size)
        self.redraw = None

        self.double_pi = 2 * math.pi

        self.action = action

        # 目标方位相关
        self.last_raw_goal_abs_dir = None  # 记录地面坐标系下目标相对于机体的方向
        self.compensate_goal_abs_dir = None  # 跨越180°线会产生不连续的角度值，通过补偿将其变成连续值
        # 目标距离相关
        self.straight_line_dis = None
        # 障碍距离相关
        div, mod = divmod(max_detect_angle, detect_angle_interval)
        assert mod == 0
        self.max_detect_angle = math.radians(max_detect_angle)
        self.detect_angle_interval = math.radians(detect_angle_interval)
        self.safe_dis = safe_dis
        self.add_noise = add_noise
        self.max_detect_range = max_detect_range
        self.no_obstacle_dis = 2 * self.max_detect_range  # 探测方向上没有障碍时的距离，需要区别于障碍位于max_detect_range处

        # 决策信息：障碍距离+目标位置
        self.obs_distances, self.goal_dis, self.goal_dir = None, None, None
        self.last_goal_dis, self.last_goal_abs_dir = None, None  # 需要加绝对值

        self.d_obstacles = 2 * div + 1  # 周围环境
        self.d_states_detail = (self.d_obstacles, 2)  # 状态构成
        self.d_states = sum(self.d_states_detail)  # 周围环境+目标方位
        self.terminal = np.zeros(self.d_states)  # 结束状态
        self.result = None
        self.residual_traj = None               # 当前位置到目标位置的剩余轨迹（直线）

        # 控制量
        self.delta_dis = None
        self.delta_dir = None
        self.delta_yaw_rate = None
        self.delta_time = None
        self.yaw_rate = None

        self.circle_obstacles = None
        self.line_obstacles = np.array([
            [0, 0, self.region_width, 0],
            [self.region_width, 0, self.region_width, self.region_height],
            [self.region_width, self.region_height, 0, self.region_height],
            [0, self.region_height, 0, 0]], dtype=np.float) if use_border else None

    def set_obstacles(self, circle_obstacles, line_obstacles):
        if circle_obstacles is not None:
            self.circle_obstacles = np.array(circle_obstacles)
        if line_obstacles is not None:
            self.line_obstacles = np.array(line_obstacles) if self.line_obstacles is None else \
                np.concatenate((self.line_obstacles, line_obstacles))

    def render_trajectory(self, points, show_scope=False, circle_scope=True, show_arrow=False, show_pos=True):
        """根据点列绘制轨迹"""
        for i in range(len(points)-1):
            self.traj.line2(points[i], points[i+1])
            self.current_pos = points[i+1]
            delta_x, delta_y = points[i+1]-points[i]
            self.current_dir = math.atan2(delta_y, delta_x)
            self.render(0., True, show_scope, circle_scope, show_arrow, show_pos)

    def render(self, sleep_time=0.01, show_trace=True, show_scope=False, circle_scope=True, show_arrow=False, show_pos=False):
        # call step before render
        if self.redraw:     # draw scenairo
            self.redraw = False
            self.canvas.create()    # create or reset canvas
            # draw start point and goal point
            radius = 8
            self.canvas.draw_oval(self.init_pos, radius, 'black', transform_radius=False)
            self.canvas.draw_oval(self.goal_pos, radius, 'yellow', transform_radius=False)
            # if show_arrow:
            self.canvas.draw_arrow(self.init_pos, self.init_dir, 8 * radius, 'black', transform_length=False)
            if circle_scope:
                self.canvas.draw_oval(self.init_pos, self.max_detect_range, 'pink', background=True)
            else:
                self.canvas.draw_sector(self.init_pos, self.max_detect_range, self.init_dir - math.pi / 2, math.pi,
                                        'pink', background=True)
            if self.circle_obstacles is not None:
                for obstacle in self.circle_obstacles:
                    self.canvas.draw_oval(obstacle[:2], obstacle[2], 'deepskyblue')
            if self.line_obstacles is not None:
                for obstacle in self.line_obstacles:
                    self.canvas.draw_line(obstacle[:2], obstacle[2:], 'deepskyblue')
        if show_trace:      # draw trace
            if self.traj.traj_type == self.traj.ARC:
                self.canvas.draw_arc(*self.traj.traj, 'black')
            else:
                self.canvas.draw_line(*self.traj.traj, 'black')
            if show_pos:
                self.canvas.draw_oval(self.current_pos, 3, 'red', transform_radius=False)
            if show_arrow:
                self.canvas.draw_arrow(self.current_pos, self.current_dir, 20, 'green', transform_length=False)
            if show_scope:
                if circle_scope:
                    self.canvas.draw_oval(self.current_pos, self.max_detect_range, 'pink', background=True)
                else:
                    self.canvas.draw_sector(self.current_pos, self.max_detect_range, self.current_dir - math.pi / 2, math.pi, 'pink', background=True)
        self.canvas.update(sleep_time)

    def reset(self, scenairo):
        self.redraw = True

        self.set_obstacles(scenairo.circle_obstacles, scenairo.line_obstacles)
        self.init(scenairo.init_pos, scenairo.init_dir, scenairo.goal_pos, scenairo.goal_dir)

        self.last_raw_goal_abs_dir = 0.
        self.compensate_goal_abs_dir = 0.

        self.delta_dis = None
        self.yaw_rate = 0.
        self.delta_yaw_rate = None
        self.delta_time = None
        self.delta_dir = None
        self.result = self.timeout
        self.residual_traj = None
        self._state()
        self.straight_line_dis = self.goal_dis
        self.last_goal_dis, self.last_goal_abs_dir = self.goal_dis, abs(self.goal_dir)
        return self.state

    def step(self, action):
        """进入吸收状态后，有些状态没有被更新，继续调用step会导致结果未定义"""
        self.action(action)
        self.delta_dir, self.delta_dis = self.action.delta_dir, self.action.delta_dis
        # 左转和右转相反，直行时半径无穷大，取极限
        self.transition(self.delta_dir, self.delta_dis)

        self.delta_time = self.delta_dis / self.action.linear_velocity

        last_yaw_rate = self.yaw_rate
        self.yaw_rate = self.delta_dir / self.delta_time
        self.delta_yaw_rate = self.yaw_rate - last_yaw_rate

        # print('delta_t: {0}, delta_d: {1}, yaw_rate: {2}, delta_yaw_rate: {3}, delta_angle: {4}'.
        #       format(self.delta_time, self.delta_dis, self.yaw_rate, self.delta_yaw_rate, self.delta_dir))

        # basic reward: failure + success
        if not self.is_safe():  # failure
            self.result = self.failure
            return self.terminal, -10., True

        self._state()
        if Geometry.distance_p2seg(self.goal_pos, self.last_pos, self.current_pos) <= \
                4. * self.action.basic_distance_step:  # success
            self.result = self.success
            self.residual_traj = self.current_pos, self.goal_pos
            return self.terminal, 10., True

        # greed reward
        reward = 0.
        if self.goal_dis < self.last_goal_dis:  # distance decrease
            reward += 0.2
        else:  # distance increase
            reward += -0.2
        goal_abs_dir = abs(self.goal_dir)
        if goal_abs_dir < self.last_goal_abs_dir:  # direction decrease
            reward += 0.2
        else:  # direction increase
            reward += -0.2

        self.last_goal_dis, self.last_goal_abs_dir = self.goal_dis, goal_abs_dir
        return self.state, reward, False

    def is_safe(self):
        if self.circle_obstacles is None:
            circle_safe = True
        else:  # 只检查了点是否落入障碍内，而没有检查线段是否穿越障碍的情况
            # square_dis = np.sum(np.square(self.circle_obstacles[:, :2] - self.current_pos), axis=1)
            # circle_safe = np.all(square_dis > np.square(self.circle_obstacles[:, 2]))

            if self.traj.traj_type == self.traj.ARC:    # 当前轨迹为弧
                for circle in self.circle_obstacles:
                    center, radius = circle[:2], circle[2]
                    dis = Geometry.distance_p2arc(center, *self.traj.traj)
                    if dis < radius:
                        return False
            else:
                for circle in self.circle_obstacles:
                    center, radius = circle[:2], circle[2]
                    dis = Geometry.distance_p2seg(center, *self.traj.traj)
                    if dis < radius:
                        return False

        if self.line_obstacles is None:
            line_safe = True
        else:
            ob0 = self.line_obstacles[:, 0]
            ob1 = self.line_obstacles[:, 1]
            ob2 = self.line_obstacles[:, 2]
            ob3 = self.line_obstacles[:, 3]
            x_ab, y_ab = ob0 - ob2, ob1 - ob3
            x_dc, y_dc = self.current_pos[0] - self.last_pos[0], self.current_pos[1] - self.last_pos[1]
            x_ac, y_ac = ob0 - self.last_pos[0], ob1 - self.last_pos[1]
            det = x_ab * y_dc - x_dc * y_ab

            mask = np.abs(det) > 1e-2
            Det = det[mask]
            X_ac = x_ac[mask]
            Y_ac = y_ac[mask]
            k1 = (y_dc * X_ac - x_dc * Y_ac) / Det
            k2 = (-y_ab[mask] * X_ac + x_ab[mask] * Y_ac) / Det
            tem1 = np.logical_and(0 <= k1, k1 <= 1)
            tem2 = np.logical_and(0 <= k2, k2 <= 1)
            line_safe = not np.any(np.logical_and(tem1, tem2))
        # return line_safe and circle_safe
        return line_safe    # and circle_safe

    def _state(self):
        all_dis = []
        if self.circle_obstacles is not None:
            all_dis.append(self.get_circle_ob_dis())
        if self.line_obstacles is not None:
            all_dis.append(self.get_line_ob_dis())
        self.obs_distances = np.min(all_dis, axis=0)
        if self.add_noise:
            # distance += 0.1 * distance * np.random.randn(distance.shape[0])
            self.obs_distances *= 1 + np.random.normal(loc=0, scale=0.05, size=(self.d_obstacles,))
        self.goal_dir, self.goal_dis = self.get_goal_dir(), self.get_goal_dis()

    @property
    def state(self):
        return np.concatenate((self.obs_distances / self.max_detect_range,
                               [self.goal_dis / self.straight_line_dis, self.goal_dir / math.pi]))

    def get_goal_dis(self):
        return Geometry.distance(self.goal_pos - self.current_pos)

    def get_goal_dir(self):
        x, y = self.goal_pos - self.current_pos  # 看作目标相对于机体移动
        raw_goal_abs_dir = np.arctan2(y, x)
        # 检测地面坐标系下的目标方向跨越180°线情况并进行相应补偿
        diff_raw_goal_abs_dir = raw_goal_abs_dir - self.last_raw_goal_abs_dir
        if diff_raw_goal_abs_dir < -math.pi:  # 逆时针跨过180°，暂定检测阈值为180°，所以要求单步动作小于180°
            self.compensate_goal_abs_dir += self.double_pi
        elif diff_raw_goal_abs_dir > math.pi:  # 顺时针跨国180°
            self.compensate_goal_abs_dir -= self.double_pi
        self.last_raw_goal_abs_dir = raw_goal_abs_dir
        real_goal_abs_dir = raw_goal_abs_dir + self.compensate_goal_abs_dir  # 补偿
        # 机体坐标系下的目标方向，目标在机体右侧为正
        goal_rel_dir = self.current_dir - real_goal_abs_dir
        if goal_rel_dir > math.pi:  #
            self.compensate_goal_abs_dir += self.double_pi
            goal_rel_dir -= self.double_pi
        if goal_rel_dir <= -math.pi:
            self.compensate_goal_abs_dir -= self.double_pi
            goal_rel_dir += self.double_pi
        return goal_rel_dir

    def get_circle_ob_dis(self):
        distance = np.full((self.d_obstacles, len(self.circle_obstacles)), self.no_obstacle_dis)
        angle = np.arange(self.max_detect_angle, -self.max_detect_angle - self.detect_angle_interval / 2,
                          -self.detect_angle_interval)
        rad_angle = self.current_dir + angle
        s_angle = np.sin(rad_angle)
        c_angle = np.cos(rad_angle)

        x1 = self.current_pos[0] - self.circle_obstacles[:, 0]
        y1 = self.current_pos[1] - self.circle_obstacles[:, 1]

        coe1 = c_angle[:, np.newaxis] * x1[np.newaxis, :] + s_angle[:, np.newaxis] * y1[np.newaxis, :]

        square_coe1 = np.square(coe1)
        coe2 = np.square(x1) + np.square(y1) - np.square(self.circle_obstacles[:, 2])

        coe = square_coe1 - coe2
        mask = coe >= 0

        dis = -np.sqrt(coe[mask]) - coe1[mask]

        dis[np.logical_or(dis < 0, dis > self.max_detect_range)] = self.no_obstacle_dis
        distance[mask] = dis
        return np.min(distance, axis=1)

    def get_line_ob_dis(self):
        """
        OA = [x1, y1]
        OB = [x2, y2]
        OC = [x3, y3]
        CD = k[x4, y4]

        OC + kCD - OA = k'(OB - OA) (k>0, 0<k'<1)
        k' = [(y1-y3)*x4 + （x3-x1)*y4] / [(x2-x1)*y4 + (y1-y2)*x4]
        """
        distance = np.full((self.d_obstacles, len(self.line_obstacles)), self.no_obstacle_dis)
        angle = np.arange(self.max_detect_angle, -self.max_detect_angle - self.detect_angle_interval / 2,
                          -self.detect_angle_interval)
        rad_angle = self.current_dir + angle
        c_angle = np.cos(rad_angle)
        c_angle2 = c_angle[:, np.newaxis]
        s_angle = np.sin(rad_angle)
        s_angle2 = s_angle[:, np.newaxis]

        ob0 = self.line_obstacles[:, 0]
        ob1 = self.line_obstacles[:, 1]
        ob2 = self.line_obstacles[:, 2]
        ob3 = self.line_obstacles[:, 3]

        diff1 = ob2 - ob0
        diff1 = diff1[np.newaxis, :]
        diff2 = ob1 - ob3
        diff2 = diff2[np.newaxis, :]
        coe2 = diff1 * s_angle2 + diff2 * c_angle2
        diff1 = ob1 - self.current_pos[1]
        diff1 = diff1[np.newaxis, :]
        diff2 = self.current_pos[0] - ob0
        diff2 = diff2[np.newaxis, :]
        coe1 = diff1 * c_angle2 + diff2 * s_angle2

        mask1 = np.abs(coe2) > 1e-2
        rate_list = coe1[mask1] / coe2[mask1]
        mask2 = np.logical_and(0 <= rate_list, rate_list <= 1)
        mask1[mask1] = mask2  # new mask1

        rate_list = rate_list[mask2]  # new list

        indices = np.nonzero(mask1)
        indice_x = indices[0]
        indice_y = indices[1]

        coe1 = ob2[indice_y] * rate_list + (1 - rate_list) * ob0[indice_y] - self.current_pos[0]
        coe1 = coe1.astype(np.int32)
        coe2 = ob3[indice_y] * rate_list + (1 - rate_list) * ob1[indice_y] - self.current_pos[1]
        coe2 = coe2.astype(np.int32)

        tst1 = coe1 * c_angle[indice_x]
        tst2 = coe2 * s_angle[indice_x]

        mask3 = np.logical_and(tst1 >= 0, tst2 >= 0)
        dis = np.sqrt(coe1[mask3] ** 2 + coe2[mask3] ** 2)

        mask4 = dis < self.max_detect_range

        mask3[mask3] = mask4
        mask1[mask1] = mask3

        distance[mask1] = dis[mask4]

        return np.min(distance, axis=1)

