import os

# import threading
#
#
# def singleton(cls):
#     instance = None
#     instance_lock = threading.Lock()
#
#     def _singleton(*args, **kwargs):
#         nonlocal instance
#         if instance is None:
#             with instance_lock:
#                 if instance is None:
#                     instance = cls(*args, **kwargs)
#         return instance
#
#     return _singleton


class Path:
    """管理目录，方便创建文件和扩展子目录"""

    def __init__(self, root_dir, create=True):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            if create:
                os.makedirs(self.root_dir)
            else:
                raise FileNotFoundError

    def __call__(self, *paths):
        """扩展root_dir"""
        return Path(self.join(*paths))

    def join(self, *paths):
        return os.path.join(self.root_dir, *paths)


class Version:
    """
    """
    def __init__(self, working_dir):
        """working_dir必须是Path对象，以保证working_dir已创建"""
        assert isinstance(working_dir, Path)
        self.__info_file = working_dir.join('version.txt')
        self.initial_version = 0
        if not os.path.exists(self.__info_file):
            self.latest_used_version = self.initial_version
        else:
            self.latest_used_version = self.load()
        self.__next_version = None

    def load(self):
        with open(self.__info_file, 'r') as fp:
            return int(fp.readline())

    def save(self, version):
        with open(self.__info_file, 'w') as fp:
            fp.write(str(version))

    @property
    def next_available_version(self):
        if self.__next_version is None:
            self.__next_version = self.latest_used_version + 1
            self.save(self.__next_version)
        return self.__next_version

    def update(self):
        self.latest_used_version = self.next_available_version
        self.__next_version = None

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def __call__(self):
        return self.__dict__
