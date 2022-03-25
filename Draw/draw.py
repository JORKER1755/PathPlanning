"""
Convert an object from a Cartesian coordinate system to tkinter's coordinate system
"""

import numpy as np
import time
import tkinter as tk
import common
import math

if common.use_windows:
    import pyautogui


class Canvas:
    """delayed creation"""
    def __init__(self, canvas_size, region_size, title='Path Planning', create=False):
        self.canvas_size = np.array(canvas_size)
        self.canvas_width, self.canvas_height = canvas_size
        self.scale = self.canvas_size/region_size       # 尺度，画布大小=区域大小*尺度
        self.title = title

        self.canvas = None
        if create: self.create()

    def create(self):
        """create or reset"""
        if self.canvas is None:
            root = tk.Tk()
            root.title(self.title)
            root.geometry('{0}x{1}+{x}+{y}'.format(self.canvas_width, self.canvas_height, x=0, y=0))
            # root.state("zoomed")
            self.canvas = tk.Canvas(root, bg='white', height=self.canvas_height, width=self.canvas_width)
            self.canvas.pack()
        else:
            self.canvas.delete('all')

    def update(self, sleep_time):
        self.canvas.update()
        time.sleep(sleep_time)

    def transform_abs_coord(self, point):
        """因为y轴转换涉及偏移，只能转换绝对位置，不支持相对位置"""
        x, y = point * self.scale  # 从region_size到canvas_size
        return np.array([x, self.canvas_height - y], dtype=np.int)

    def transform_rel_coord(self, direct, scale=True):
        """不负责将数据类型转为int"""
        x, y = direct * self.scale if scale else direct
        return np.array([x, -y])

    def draw_oval(self, center, radius, color, transform_radius=True, background=False):
        """transform_radius为False，要求radius为int类型"""
        if transform_radius:
            lowerleft, topright = self.transform_abs_coord(center - radius), self.transform_abs_coord(center + radius)
        else:
            center = self.transform_abs_coord(center)
            lowerleft, topright = center - radius, center + radius
        item = self.canvas.create_oval(*lowerleft, *topright, fill=color, outline='')
        if background: self.canvas.lower(item)

    def draw_line(self, p1, p2, color):
        self.canvas.create_line(*self.transform_abs_coord(p1), *self.transform_abs_coord(p2), fill=color)

    def draw_arrow(self, start, direction, length, color, transform_length=True):
        """transform_length为False，要求length为int类型"""
        start = self.transform_abs_coord(start)
        direction = np.array([np.cos(direction), np.sin(direction)])
        end = start + (length * self.transform_rel_coord(direction, scale=transform_length)).astype(np.int)
        self.canvas.create_line(*start, *end, arrow=tk.LAST, fill=color)

    def draw_sector(self, center, radius, start, extent, color, background=False):
        """theta为弧度制，不区分起点和终点，属于(-pi, pi]"""
        center = self.transform_abs_coord(center)
        lefttop = center - [radius, radius]
        rightlower = center + [radius, radius]
        item = self.canvas.create_arc(tuple(np.concatenate((lefttop, rightlower))),
                                      start=math.degrees(start), extent=math.degrees(extent), style=tk.PIESLICE, outline='', fill=color)
        if background: self.canvas.lower(item)

    def draw_arc(self, center, radius, start, extent, color):
        """theta为弧度制，不区分起点和终点，属于(-pi, pi]"""
        center = self.transform_abs_coord(center)
        lefttop = center - [radius, radius]
        rightlower = center + [radius, radius]
        self.canvas.create_arc(tuple(np.concatenate((lefttop, rightlower))),
                               start=math.degrees(start), extent=math.degrees(extent), style=tk.ARC, fill=color)

    def screenshot(self, path='img'):
        gap = np.array([12, 2])  # 左上角存在小缝隙
        region = (*gap, *np.array([self.canvas_width, self.canvas_height + 40]))  # 窗口大小与画布大小是不同的
        img = pyautogui.screenshot(region=region)
        img.save(path + '.png')


if __name__ == '__main__':
    c = Canvas((500, 500), (500, 500), create=True)
    # c.draw_line([200, 200, 300, 300], 'red')
    # c.draw_arc2([400,  300], 100, 0, -90)
    c.update(0.1)
    while True:
        pass