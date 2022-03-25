import numpy as np


class ReplayMemory:
    """It must be ensured that the data in ReplayMemory will not be modified externally, be careful with in-place operations"""
    def __init__(self, buffer_size, state_dt=np.ndarray, action_dt=np.int, **kwargs):
        self.exp_dt = [('s', state_dt), ('a', action_dt), ('r', np.float), ('s_', state_dt), ('d', np.bool)]
        self.buffer_size = buffer_size
        self.buffer = np.empty(self.buffer_size, dtype=self.exp_dt)
        self.counter = 0

    def append(self, exp):
        """exp: (s, a, r, s_, done)"""
        self.buffer[self.counter % self.buffer_size] = exp
        self.counter += 1

    def sample(self, batch_size):
        """return: s, a, r, s_, done"""
        if self.counter >= self.buffer_size:  # 已满
            sample_index = np.random.choice(self.buffer_size, size=batch_size)
        else:  # 未满
            sample_index = np.random.choice(self.counter, size=batch_size)
        batch_exp = self.buffer[sample_index]
        return self.unzip(batch_exp)

    @staticmethod
    def unzip(non_nested):
        """transpose operation or inverse operation of zip"""
        return tuple(map(np.array, zip(*non_nested)))


if __name__ == '__main__':
    buf = ReplayMemory(4)
    for i in range(6):
        obs = np.array([[2., 3., 5.]])
        buf.append((obs, 2, 2.2, obs, False))  # 必须是tuple
    # print(buf.buffer)
    s, a, r, s_, d = buf.sample(3)
    print(s, a, r, s_, d, sep='\n')
