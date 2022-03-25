"""
将工程所在目录添加到环境变量PATH中，将python包的导入路径与文件的访问路径统一为绝对路径，使得IDE下和命令行下均可执行本工程
**此文件必须在工程根目录下**
"""

import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from utils import Path
project_dir = Path(project_dir)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Used to open/close GUI related code
use_windows = sys.platform == 'win32'
