import tensorflow as tf


def model(cls):
    def _model(*args, **kwargs):
        """kwargs仅包含实际提供的，不包含默认参数"""
        """实例化子类时cls为子类对象，实例化本类时cls为本类对象"""
        scope_name = ('eval_' if kwargs['eval_model'] else 'target_') + cls.__name__
        with tf.variable_scope(scope_name):
            self = cls(*args, **kwargs)
        setattr(self, 'vars', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name))  # 所有weight
        return self
    return _model


def shared(cls):
    """借助单例模式实现模型共享"""
    _instance = {}

    def _shared(eval_model: bool, *args, **kwargs):
        """eval_model用于区别Model和TargetModel，使得它们不共用被共享的模型"""
        nonlocal _instance
        if eval_model not in _instance:
            # print(shared_id)
            _instance[eval_model] = cls(eval_model, *args, **kwargs)
            # print(_instance)
        return _instance[eval_model]
    return _shared
