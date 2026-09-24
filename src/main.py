"""EO-Sim-NSGA-II 整体流程入口（两阶段串联）。

用法：
    from src.main import load_instance, run
    inst = load_instance("dataset/10-5.xlsx")
    alg = run(inst.processing_times, inst.priorities, inst.due_dates,
              population_size=100, num_generations=100, use_eo=True)

两步流程（对应用户要求）：
    1. 生成初始解——use_eo=True 时先跑 EO（单目标确定性 makespan，max_iter 默认 100），
       整群解码为 (OS, MS) 染色体作 NSGA-II 初始种群（EO-Sim-NSGA-II）；
       use_eo=False 时随机初始化（基线 Sim-NSGA-II）。
    2. 调用 SimNSGAII.optimize() 演化并返回结果对象（population / pareto_front / history）。

xlsx 算例格式（dataset/*.xlsx，已实测核实）：
    - 第 1 个 sheet：i(0,1)=Number of Jobs、i(1,1)=Number of Machines；
    - "Processing Time"：行索引 J1..JN、列表头 O1..OM 的 J×M 加工时间矩阵；
    - "Machines Sequence"：不用（MS 为决策变量参与进化）；
    - "Priority and Due date"：两列 priority / due date。
"""
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from src.eo.equilibrium_optimizer import EquilibriumOptimizer
from src.nsga2.algorithm import SimNSGAII


class Instance(NamedTuple):
    """一个 FJSSP 算例的静态数据（不含决策变量）。"""
    num_jobs: int
    num_machines: int
    processing_times: np.ndarray    # (J, M)
    priorities: np.ndarray         # (J,)，正数
    due_dates: np.ndarray         # (J,)


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


def run(processing_times, priorities, due_dates,
         population_size: int, num_generations: int, *,
         use_eo: bool = True, eo_max_iter: int = 100,
         crossover_rate: float = 0.9, mutation_rate: float = 0.1,
         uncertain_level: float = 0.1, sim: bool = True,
         s_short: int = 20, s_long: int = 1000,
         rng: Optional[np.random.Generator] = None,
         verbose: bool = True) -> SimNSGAII:
    """整体流程入口：生成初始解（用/不用 EO）→ SimNSGAII.optimize() → 返回结果对象。

    population_size 与 num_generations 为必填（论文 §5 参数表未读，不给默认值）；
    eo_max_iter 默认 100（CLAUDE.md 项目约定）。单个 rng 顺序共享两阶段，
    同一种子下整条流程可复现。
    """
    pt = np.asarray(processing_times, dtype=float)
    num_jobs, num_machines = pt.shape
    if rng is None:
        rng = np.random.default_rng()

    # ---- 步骤 1：生成初始解（EO 整群解码 / 随机初始化） ----
    initial_population = None
    if use_eo:
        eo = EquilibriumOptimizer(num_jobs, num_machines, pt,
                                   population_size=population_size,
                                   max_iter=eo_max_iter, rng=rng).optimize()
        initial_population = eo.chromosomes()
        if verbose:
            print(f"[EO] {eo_max_iter} 代最优确定性 makespan = {eo.best_fitness:.4f}")

    # ---- 步骤 2：调用 optimize 获取最终解 ----
    alg = SimNSGAII(num_jobs, num_machines, pt, priorities, due_dates,
                    population_size=population_size,
                    num_generations=num_generations,
                    crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                    uncertain_level=uncertain_level, sim=sim,
                    s_short=s_short, s_long=s_long,
                    initial_population=initial_population, rng=rng).optimize()

    if verbose:
        print(f"[NSGA-II] 每代 F0 规模 history = {alg.history}")
        print(f"[结果] 最终 Pareto 前沿 {len(alg.pareto_front)} 个成员"
              "（长模拟重评价后）:")
        for i, c in enumerate(alg.pareto_front):
            print(f"  #{i + 1}: makespan = {c.makespan:.4f}, twte = {c.twte:.4f}")
    return alg


if __name__ == "__main__":
    # 1. load_instance：读 dataset/*.xlsx
    inst = load_instance("../dataset/10-5.xlsx")
    # 2. use_eo=True
    alg = run(inst.processing_times, inst.priorities, inst.due_dates,
              population_size=100, num_generations=400, use_eo=True,
              eo_max_iter=100, crossover_rate=0.8, mutation_rate=0.2, uncertain_level=0.5, sim=False, s_short=20, s_long=10000, rng=np.random.default_rng(1))