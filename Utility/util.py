import os


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
    def __init__(self, working_dir, init=0):
        """本地保存下一可用(未被使用)的版本号, working_dir必须是Path对象，以保证working_dir已创建"""
        assert isinstance(working_dir, Path)
        self.info_file = working_dir.join('version.txt')
        if not os.path.exists(self.info_file):
            self.version = init
        else:
            with open(self.info_file, 'r') as fp:
                self.version = int(fp.readline())

    @property
    def latest_version_str(self):
        return str(self.version-1)

    @property
    def version_str(self):
        return str(self.version)

    @property
    def version_plusone_str(self):
        return str(self.version+1)

    def update(self):
        self.version += 1
        with open(self.info_file, 'w') as fp:
            fp.write(self.version_str)

    @property
    def version_str_plusplus(self):
        """模拟v++"""
        v_str = self.version_str
        self.update()
        return v_str

    @property
    def plusplus_version_str(self):
        """模拟++v"""
        self.update()
        return self.version_str


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def __call__(self):
        return self.__dict__
