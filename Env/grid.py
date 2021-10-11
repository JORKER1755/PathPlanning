import numpy as np
import time
import tkinter as tk
import common
if common.use_windows:
    import pyautogui    # 禁止窗口的放大


class Grid:
    """通常的直角坐标系：x轴水平向右，y轴竖直向上，位置坐标：(x,y)"""
    canvas_width, canvas_height = 700, 700
    timeout, failure, border, success, back, n_outcomes = 0, 1, 2, 3, 4, 5
    UP, TOPLEFT, LEFT, LOWERLEFT, DOWN, LOWERRIGHT, RIGHT, TOPRIGHT = np.arange(8)

    def __init__(self):
        self.width_size, self.height_size = None, None
        self.scale = None

        self.init_pos = None
        self.goal_pos = None
        self.current_pos = None
        self.scene = None

        self.n_actions = 8
        self.actions = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]], dtype=np.int32)

        self.result = None
        self.max_reward = 10
        self.rewards = np.array([1, 1.414, 1, 1.414, 1, 1.414, 1, 1.414])

        self.canvas = None
        self.redraw = None
        self.current_draw = None

    def set_task(self, init_pos, goal_pos):
        self.init_pos = np.array(init_pos)
        self.goal_pos = np.array(goal_pos)
        return self

    def set_scene(self, scene):
        self.scene = np.array(scene)
        self.width_size, self.height_size = self.scene.shape
        self.scale = np.array([self.canvas_width / self.width_size, self.canvas_height / self.height_size])
        return self

    def reset(self):
        self.redraw = True
        self.current_draw = None

        self.current_pos = self.init_pos.copy()
        self.result = 0
        return self.current_pos

    def step(self, action):
        last_pos = self.current_pos.copy()
        self.current_pos += self.actions[action]

        x, y = self.current_pos
        if 0 <= x < self.width_size and 0 <= y < self.height_size:
            if self.scene[x][y] == 1:
                self.result = self.failure
                return self.current_pos, -5, True
            if np.all(self.current_pos == self.goal_pos):
                self.result = self.success
                reward = -0.1 * self.rewards[action]  # 统一 reward（后面实现全部统一）
                return self.current_pos, reward, True
            if np.all(self.current_pos == self.init_pos):
                self.result = self.back
                return self.current_pos, -self.max_reward, True
            reward = -0.1 * self.rewards[action]
            return self.current_pos, reward, False
        else:
            self.result = self.border
            self.current_pos = last_pos     # 复原
            return last_pos, -self.max_reward, False

    def render(self, sleep_time=0.01):
        # func step before render
        if self.redraw:
            self.redraw = False
            if self.canvas is None:
                root = tk.Tk()
                root.title('Grid World')
                root.geometry('{0}x{1}+{x}+{y}'.format(self.canvas_width, self.canvas_height, x=0, y=0))
                # root.state("zoomed")
                self.canvas = tk.Canvas(root, bg='white', height=self.canvas_height, width=self.canvas_width)
                self.canvas.pack()
            self.canvas.delete('all')
            self.draw_task()
        if self.current_draw is not None:   # 起点不需要删除，第二步开始
            self.canvas.delete(self.current_draw)
        self.current_draw = self.draw_oval(self.current_pos, 0.25, 'red')
        self.canvas.update()
        time.sleep(sleep_time)

    def draw_task(self):
        radius = 0.25   # 四分之一Grid
        self.draw_oval(self.init_pos, radius, 'black')
        self.draw_oval(self.goal_pos, radius, 'yellow')

        for x in range(self.width_size):
            for y in range(self.height_size):
                if self.scene[x, y] == 1:
                    self.draw_rectangle(np.array([x, y]), color='black')

    def draw_rectangle(self, pos, color):
        self.canvas.create_rectangle(*self.transform_abs_coord(pos), *self.transform_abs_coord(pos + [1, 1]),
                                     fill=color)

    def draw_oval(self, center, radius, color, transform_radius=True):
        """transform_radius为False，要求radius为int类型"""
        center = center + [0.5, 0.5]    # 从左下角调整到中心
        if transform_radius:
            lowerleft, topright = self.transform_abs_coord(center - radius), self.transform_abs_coord(center + radius)
        else:
            center = self.transform_abs_coord(center)
            lowerleft, topright = center - radius, center + radius
        return self.canvas.create_oval(*lowerleft, *topright, fill=color)

    def transform_abs_coord(self, point):
        """因为y轴转换涉及偏移，只能转换绝对位置，不支持相对位置"""
        x, y = point * self.scale  # 从region_size到canvas_size
        return np.array([x, self.canvas_height - y], dtype=np.int)


if __name__ == '__main__':
    s = np.zeros((20, 20))
    s[2, 5] = 1
    env = Grid().set_task([10, 10], [1, 1]).set_scene(s)
    # [[0, 0, 0, 0, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 1, 0, 1, 0],
    #  [0, 0, 1, 1, 1, 0, 1, 0],
    #  [0, 0, 1, 0, 0, 0, 1, 0],
    #  [0, 0, 1, 0, 1, 1, 1, 0],
    #  [0, 0, 1, 0, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 1, 0, 0, 0]
    #  ]
    env.reset()
    for _ in range(100):
        a = np.random.randint(0, 8)
        print(a)
        env.step(a)
        env.render(.3)
    print("end")
    while True:
        time.sleep(1999)
