# utils/__init__.py

# 导入模型定义
from .train import *


# 可选：定义一个__all__列表，以指定可以通过from models import * 导入的模块
# 这个列表包含了您想要暴露的模型名称
__all__ = ['predict', 'train', 'device']
