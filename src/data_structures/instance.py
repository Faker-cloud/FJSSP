"""FJSSP 算例实例数据结构：Instance 命名元组（加载后的静态数据，不含决策变量）。"""
from typing import NamedTuple

import numpy as np


class Instance(NamedTuple):
    """一个 FJSSP 算例的静态数据（不含决策变量）。"""
    num_jobs: int
    num_machines: int
    processing_times: np.ndarray    # (J, M)
    priorities: np.ndarray         # (J,)，正数
    due_dates: np.ndarray         # (J,)