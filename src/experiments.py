"""实验层：批量运行 + 论文 §5.1 口径统计 + xlsx 导出。

对每个算例、每档不确定性水平（论文 §5.1：UL = 0.5 / 1.0 / 2.0）、两种算法
（EO-Sim-NSGA-II / 基线 Sim-NSGA-II）各跑 30 次，统计：
    - f_s：两目标各自取 30 次运行中最优（每次运行取其长模拟 Pareto 前沿上的目标极值）；
    - f_d：每算例跑 2 次确定性版本（sim=False）——确定性 EO-NSGA-II（EO 行的分母）
      与确定性 NSGA-II（基线行的分母），即每个随机版本与自己的确定性对照比 gap；
    - gap（论文 Eq.15）：Δ_o = (f_s − f_d) / f_d，导出为小数比例（非百分数）；
    - NPS（Pareto 解个数）与 CPU Time（仅计 NSGA-II 演化阶段，EO 初始化不计）取均值。

结果导出为单 sheet 汇总表（tidy，每行 = 算例 × UL × 算法）。
参数默认值取自论文 Table 2（N=100、G=400、cr=0.8、mr=0.2、EO 100 代、短 20 / 长 10000），
均可覆盖。种子由 base_seed 与 (算例, UL, 算法, 运行序号) 派生，各组合独立可复现。
"""

import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eo.equilibrium_optimizer import EquilibriumOptimizer
from src.main import Instance, load_instance
from src.nsga2.algorithm import SimNSGAII

ALGORITHMS = ("EO-Sim-NSGA-II", "Sim-NSGA-II")   # 下标 0 = EO 版，1 = 基线

_COLUMNS = ["instance", "num_jobs", "num_machines", "uncertain_level", "algorithm",
            "f_s_makespan", "f_s_twte", "f_d_makespan", "f_d_twte",
            "gap_makespan", "gap_twte", "nps_mean", "cpu_time_mean"]


def _run_once(inst: Instance, *, use_eo: bool, sim: bool, uncertain_level: float,
              population_size: int, num_generations: int, crossover_rate: float,
              mutation_rate: float, eo_max_iter: int, s_short: int, s_long: int,
              rng: np.random.Generator):
    """单次完整演化：EO 初始化（可选，不计时）+ NSGA-II（计时）。

    返回 (SimNSGAII 结果对象, NSGA-II 阶段耗时秒)。CPU Time 口径仅含 NSGA-II 演化。
    """
    initial_population = None
    if use_eo:
        eo = EquilibriumOptimizer(inst.num_jobs, inst.num_machines,
                                  inst.processing_times,
                                  population_size=population_size,
                                  max_iter=eo_max_iter, rng=rng).optimize()
        initial_population = eo.chromosomes()

    t0 = time.perf_counter()
    alg = SimNSGAII(inst.num_jobs, inst.num_machines, inst.processing_times,
                    inst.priorities, inst.due_dates,
                    population_size=population_size,
                    num_generations=num_generations,
                    crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                    uncertain_level=uncertain_level, sim=sim,
                    s_short=s_short, s_long=s_long,
                    initial_population=initial_population, rng=rng).optimize()
    return alg, time.perf_counter() - t0


def _front_extremes(alg: SimNSGAII):
    """长模拟重评价后前沿上两目标各自的最优值（前沿含种群各目标极值点）。"""
    return (min(c.makespan for c in alg.pareto_front),
            min(c.twte for c in alg.pareto_front))


def run_experiments(instance_paths: Sequence[str],
                    population_size: int = 100, num_generations: int = 400,
                    crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                    eo_max_iter: int = 100, s_short: int = 20, s_long: int = 10000,
                    uncertainty_levels: Sequence[float] = (0.5, 1.0, 2.0),
                    num_runs: int = 30, base_seed: int = 0,
                    output_path: str = "results/experiments.xlsx",
                    verbose: bool = True) -> pd.DataFrame:
    """批量实验入口：逐算例统计 f_s / f_d / gap / NPS / CPU，导出 xlsx 并返回汇总表。"""
    rows: List[dict] = []
    for inst_idx, path in enumerate(instance_paths):
        inst = load_instance(path)
        name = Path(path).stem

        # ---- f_d ×2：每种算法各一次确定性版本（UL 与确定性无关） ----
        f_d = {}
        for algo_idx, algo_name in enumerate(ALGORITHMS):
            rng = np.random.default_rng([base_seed, inst_idx, algo_idx])
            alg_det, _ = _run_once(
                inst, use_eo=(algo_idx == 0), sim=False, uncertain_level=0.0,
                population_size=population_size, num_generations=num_generations,
                crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                eo_max_iter=eo_max_iter, s_short=s_short, s_long=s_long, rng=rng)
            f_d[algo_name] = _front_extremes(alg_det)

        # ---- 随机版：每档 UL × 每算法 × num_runs 次 ----
        for ul_idx, ul in enumerate(uncertainty_levels):
            for algo_idx, algo_name in enumerate(ALGORITHMS):
                best_mk, best_tw = np.inf, np.inf
                nps_sum, cpu_sum = 0, 0.0
                for run_idx in range(num_runs):
                    rng = np.random.default_rng(
                        [base_seed, inst_idx, ul_idx, algo_idx, run_idx])
                    alg, cpu = _run_once(
                        inst, use_eo=(algo_idx == 0), sim=True, uncertain_level=ul,
                        population_size=population_size,
                        num_generations=num_generations,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate, eo_max_iter=eo_max_iter,
                        s_short=s_short, s_long=s_long, rng=rng)
                    mk, tw = _front_extremes(alg)
                    best_mk = min(best_mk, mk)
                    best_tw = min(best_tw, tw)
                    nps_sum += len(alg.pareto_front)
                    cpu_sum += cpu

                fd_mk, fd_tw = f_d[algo_name]
                rows.append({
                    "instance": name,
                    "num_jobs": inst.num_jobs,
                    "num_machines": inst.num_machines,
                    "uncertain_level": ul,
                    "algorithm": algo_name,
                    "f_s_makespan": best_mk,
                    "f_s_twte": best_tw,
                    "f_d_makespan": fd_mk,
                    "f_d_twte": fd_tw,
                    "gap_makespan": (best_mk - fd_mk) / fd_mk,
                    "gap_twte": (best_tw - fd_tw) / fd_tw,
                    "nps_mean": nps_sum / num_runs,
                    "cpu_time_mean": cpu_sum / num_runs,
                })
                if verbose:
                    row = rows[-1]
                    print(f"[{name} | UL={ul} | {algo_name}] "
                          f"f_s=({row['f_s_makespan']:.2f}, {row['f_s_twte']:.2f}), "
                          f"gap=({row['gap_makespan']:.4f}, "
                          f"{row['gap_twte']:.4f}), "
                          f"NPS_mean={row['nps_mean']:.1f}, "
                          f"CPU_mean={row['cpu_time_mean']:.3f}s")

    df = pd.DataFrame(rows, columns=_COLUMNS)
    out = Path(output_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    if verbose:
        print(f"已导出 {len(df)} 行 → {out}")
    return df


if __name__ == "__main__":
    run_experiments(instance_paths=("../dataset/10-5.xlsx", "../dataset/10-10.xlsx"), population_size=10, num_generations=5, eo_max_iter=10, s_long=10, num_runs=2, base_seed=7)