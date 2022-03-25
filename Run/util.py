import numpy as np
import pickle


class Record:
    @staticmethod
    def get_record_attr(name):
        def _get(self): return self.record[self.count][name]

        return _get

    @staticmethod
    def set_record_attr(name):
        def _set(self, value): self.record[self.count][name] = value

        return _set

    def __init__(self, path, size=None, items=None):
        self.path = path
        if items is None:       # 尝试本地加载
            self.__load()
        else:
            self.items = items  # {name: dtype, ...}
            self.count = 0
            self.record = np.empty(size, list(self.items.items()))          # 会自动初始化
        # 转为property
        for name in self.items:  # 创建属性对应到record的每项的get/set方法
            setattr(self.__class__, name, property(self.get_record_attr(name), self.set_record_attr(name)))

    # def __enter__(self):
    #     self.init()
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.inc()

    def init(self):
        """对需要累积的变量进行初始化，为简单这里暂时对所有的int/float进行初始化，更好方法是额外提供字典{name: init_value}"""
        for name, dtype in self.items.items():     # 初始化，仅对当前count
            setattr(self, name, dtype(0))               # dtype(0)不支持Record

    def inc(self):
        self.count += 1

    def __load(self):
        with open(self.path, 'rb') as fp:
            self.items, self.record, self.count = pickle.load(fp)

    def save(self):
        """保存items和record"""
        with open(self.path, 'wb') as fp:
            pickle.dump((self.items, self.record, self.count), fp)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [self.record[i][item] for i in range(self.count)]
        else:
            return self.record[item]

    def __len__(self):
        return self.count

    def __str__(self):
        return self.items.__str__() + '\n' + self.record.__str__()

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.count:
            item = self.record[self.iter]
            self.iter += 1
            return item
        else:
            raise StopIteration
