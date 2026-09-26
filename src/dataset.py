"""FJSSP 数据集读取：把盘上的 xlsx 算例文件加载成 In-memory 的 Instance。

xlsx 算例格式（dataset/*.xlsx，已实测核实）：
    - 第 1 个 sheet：i(0,1)=Number of Jobs、i(1,1)=Number of Machines；
    - "Processing Time"：行索引 J1..JN、列表头 O1..OM 的 J×M 加工时间矩阵；
    - "Machines Sequence"：不用（MS 为决策变量参与进化）；
    - "Priority and Due date"：两列 priority / due date。
"""
import numpy as np
import pandas as pd

from src.data_structures.instance import Instance


def load_instance(path: str) -> Instance:
    """从给定路径读取 xlsx 算例，返回 Instance（加工时间矩阵、优先级、交期）。"""
    meta = pd.read_excel(path, sheet_name=0, header=None)
    num_jobs = int(meta.iat[0, 1])
    num_machines = int(meta.iat[1, 1])

    processing_times = pd.read_excel(path, sheet_name="Processing Time",
                                     index_col=0).to_numpy(dtype=float)
    pd_data = pd.read_excel(path, sheet_name="Priority and Due date", index_col=0)
    priorities = pd_data.iloc[:, 0].to_numpy(dtype=float)
    due_dates = pd_data.iloc[:, 1].to_numpy(dtype=float)

    if processing_times.shape != (num_jobs, num_machines):
        raise ValueError(
            f"{path}: Processing Time 形状应为 {(num_jobs, num_machines)}，"
            f"实际为 {processing_times.shape}")
    if priorities.shape != (num_jobs,) or due_dates.shape != (num_jobs,):
        raise ValueError(
            f"{path}: priorities/due_dates 应为长度 {num_jobs} 的一维数组，"
            f"实际为 {priorities.shape} 与 {due_dates.shape}")
    if np.any(priorities <= 0):
        raise ValueError(f"{path}: 优先级必须为正数（提前权重为其倒数）")
    return Instance(num_jobs, num_machines, processing_times, priorities, due_dates)