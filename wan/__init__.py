#__init__.py 的作用：声明一个目录是 Python 包
from . import configs, distributed, modules
from .first_last_frame2video import WanFLF2V
from .image2video import WanI2V
from .text2video import WanT2V
from .vace import WanVace, WanVaceMP
