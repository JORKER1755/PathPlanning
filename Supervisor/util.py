

class ModelInfo:
    """将字典转为属性"""
    def __init__(self, attr_dict):
        for name, value in attr_dict.items():
            setattr(self, name, value)
