from Env.flight import Flight


class DrawScene:
    def __init__(self):
        self.draw = Flight()
        self.has_task = False

    def render(self, delay):
        if not self.has_task:
            self.set_task((0, 0), (self.draw.region_width, self.draw.region_height))
            self.has_task = True
        self.draw.redraw = True
        self.draw.render(delay, show_trace=False)

    def set_task(self, init_pos, goal_pos, init_dir=0.):
        self.draw.set_task(init_pos, goal_pos, init_dir)
        self.has_task = True

    def set_obstacles(self, circle_obstacles=None, line_obstacles=None):
        self.draw.set_obstacles(circle_obstacles, line_obstacles)

    def draw_task(self, init_pos, goal_pos, init_dir=0., delay=1.):
        self.set_task(init_pos, goal_pos, init_dir)
        self.render(delay)

    def draw_obstacle(self, circle_obstacles=None, line_obstacles=None, delay=1.):
        self.set_obstacles(circle_obstacles, line_obstacles)
        self.render(delay)
