"""实验层：批量运行 + 论文 §5.1 口径统计 + xlsx 导出。

对每个算例、每档不确定性水平（论文 §5.1：UL = 0.5 / 1.0 / 2.0）、两种算法
（EO-Sim-NSGA-II / 基线 Sim-NSGA-II）各跑 30 次，统计：
    - f_s：两目标各自取 30 次运行中最优（每次运行取其长模拟 Pareto 前沿上的目标极值）；
    - f_d：每算例跑 2 次确定性版本（sim=False）——确定性 EO-NSGA-II（EO 行的分母）
      与确定性 NSGA-II（基线行的分母），即每个随机版本与自己的确定性对照比 gap；
    - gap（论文 Eq.15）：Δ_o = (f_s − f_d) / f_d，导出为小数比例（非百分数）；
    - NPS（Pareto 解个数）与 CPU Time（仅计 NSGA-II 演化阶段，EO 初始化不计）取均值；
      NPS 为平均前沿解个数，取均值后四舍五入取整。
    - Wilcoxon 检验（论文 §5.2 / Table 4）：对每 (算例, UL)，用逐 run 前沿极值的配对
      符号秩检验比较两算法，输出两目标的 p 值与 R（+ / − / =），写入 wilcoxon sheet。

结果导出为同一 xlsx 的两个 sheet：summary（tidy，每行 = 算例 × UL × 算法）与
wilcoxon（每行 = 算例 × UL，含两目标的 p 值与 R）。
参数默认值取自论文 Table 2（N=100、G=400、cr=0.8、mr=0.2、EO 100 代、短 20 / 长 10000），
均可覆盖。随机版种子为**配对方案**（公共随机数）：每个 (算例, 运行) 内两算法的
NSGA-II 演化用同一种子流 `[base_seed, 算例, 运行, 0]`（MC 场景、锦标赛、交叉变异
抽样序列一致，配对比较），EO 阶段独立流 `[base_seed, 算例, 运行, 1]`；不同运行种子不同，
且**种子不含 UL**——同一数据集下不同 UL 水平的同一 run 复用同一种子。
figures=True 时每个 run 生成三张两算法对比图（Pareto / makespan 收敛 / twte 收敛，
见 src/plotting.py），并在全部算例跑完后按同一份 df 另画两张跨算例 gap 箱线图，
与统计共用同一次演化，不重复计算。

正式启动入口为 `launch_experiments()`（跑全量 dataset、UL=0.1/0.25/0.5、30 次，
UL 系用户指定、与论文 §5.1 的 0.5/1.0/2.0 不同）；`run_experiments` 仍可独立调用以
指定任意算例子集或覆盖参数。
"""

import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.eo.equilibrium_optimizer import EquilibriumOptimizer
from src.main import Instance, load_instance
from src.nsga2.algorithm import SimNSGAII
from src.plotting import save_run_figures, save_gap_boxplots, save_gantt

ALGORITHMS = ("EO-Sim-NSGA-II", "Sim-NSGA-II")   # 下标 0 = EO 版，1 = 基线

_COLUMNS = ["instance", "num_jobs", "num_machines", "uncertain_level", "algorithm",
            "f_s_makespan", "f_s_twte", "f_d_makespan", "f_d_twte",
            "gap_makespan", "gap_twte", "nps_mean", "cpu_time_mean"]

_TEST_COLUMNS = ["instance", "uncertain_level",
                 "p_makespan", "r_makespan", "p_twte", "r_twte"]


def _run_once(inst: Instance, *, use_eo: bool, sim: bool, uncertain_level: float,
              population_size: int, num_generations: int, crossover_rate: float,
              mutation_rate: float, eo_max_iter: int, s_short: int, s_long: int,
              rng: np.random.Generator,
              eo_rng: Optional[np.random.Generator] = None):
    """单次完整演化：EO 初始化（可选，不计时）+ NSGA-II（计时）。

    返回 (SimNSGAII 结果对象, NSGA-II 阶段耗时秒)。CPU Time 口径仅含 NSGA-II 演化。
    提供 eo_rng 时 EO 阶段用它、NSGA-II 阶段仍用 rng——两阶段随机流独立
    （公共随机数配对：两算法的 NSGA-II 演化可拿到完全相同的流）；不传则共用 rng。
    """
    initial_population = None
    if use_eo:
        eo = EquilibriumOptimizer(inst.num_jobs, inst.num_machines,
                                  inst.processing_times,
                                  population_size=population_size,
                                  max_iter=eo_max_iter,
                                  rng=rng if eo_rng is None else eo_rng).optimize()
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


def _wilcoxon_signed_rank(eo, base, alpha: float = 0.05):
    """配对 Wilcoxon 符号秩检验（精确法，双尾），返回 (p_value, R)。

    R ∈ {'+', '−', '='}：p < alpha 时按中位数方向判定（两目标均最小化，中位数小者更优）
    EO 更优记 '+'、基线更优记 '−'；p ≥ alpha 记 '='。精确法不处理零差，先丢弃；
    全零差返回 (1.0, '=')。
    """
    x = np.asarray(eo, dtype=float)
    y = np.asarray(base, dtype=float)
    keep = (x - y) != 0.0
    xf, yf = x[keep], y[keep]
    if xf.size == 0:
        return 1.0, "="
    _, p = wilcoxon(xf, yf, method="exact")
    if p >= alpha:
        return p, "="
    return p, ("+" if np.median(x) < np.median(y) else "-")


def run_experiments(instance_paths: Sequence[str],
                    population_size: int = 100, num_generations: int = 400,
                    crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                    eo_max_iter: int = 100, s_short: int = 20, s_long: int = 10000,
                    uncertainty_levels: Sequence[float] = (0.5, 1.0, 2.0),
                    num_runs: int = 30, base_seed: int = 0,
                    output_path: str = "results/experiments.xlsx",
                    figures: bool = True, figures_dir: str = "results/figures",
                    verbose: bool = True) -> pd.DataFrame:
    """批量实验入口：逐算例统计 f_s / f_d / gap / NPS / CPU，导出 xlsx 并返回汇总表。

    figures=True 时每 (算例, UL, run) 另存三张两算法对比图到
    {figures_dir}/{算例}/UL_{ul}/run_{r:02d}/，与统计共用同一次演化。
    """
    rows: List[dict] = []
    test_rows: List[dict] = []
    for inst_idx, path in enumerate(instance_paths):
        inst = load_instance(path)
        name = Path(path).stem
        t_inst = time.perf_counter()   # 计时起点：覆盖本算例 f_d + 全部 UL×run

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

        # ---- 随机版：每档 UL × num_runs 次，每 run 两算法配对运行 ----
        # 同一 run 内两算法的 NSGA-II 演化共用同一种子流（公共随机数配对比较），
        # EO 阶段用独立子流；图与统计共用同一次演化，不重复计算。
        for ul_idx, ul in enumerate(uncertainty_levels):
            stats = {algo: {"best_mk": np.inf, "best_tw": np.inf,
                            "nps": 0, "cpu": 0.0} for algo in ALGORITHMS}
            samples = {algo: {"mk": [], "tw": []} for algo in ALGORITHMS}
            for run_idx in range(num_runs):
                # 种子不含 ul_idx：同一数据集下不同 UL 水平的同一 run 用同一种子
                nsga_seed = [base_seed, inst_idx, run_idx, 0]
                eo_seed = [base_seed, inst_idx, run_idx, 1]
                results = {}
                for algo_idx, algo_name in enumerate(ALGORITHMS):
                    if verbose:
                        print(f"[{name} | UL={ul}] run {run_idx + 1}/{num_runs}: {algo_name}")
                    alg, cpu = _run_once(
                        inst, use_eo=(algo_idx == 0), sim=True, uncertain_level=ul,
                        population_size=population_size,
                        num_generations=num_generations,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate, eo_max_iter=eo_max_iter,
                        s_short=s_short, s_long=s_long,
                        rng=np.random.default_rng(nsga_seed),
                        eo_rng=(np.random.default_rng(eo_seed)
                                if algo_idx == 0 else None))
                    results[algo_name] = alg
                    mk, tw = _front_extremes(alg)
                    samples[algo_name]["mk"].append(mk)
                    samples[algo_name]["tw"].append(tw)
                    st = stats[algo_name]
                    st["best_mk"] = min(st["best_mk"], mk)
                    st["best_tw"] = min(st["best_tw"], tw)
                    st["nps"] += len(alg.pareto_front)
                    st["cpu"] += cpu
                if figures:
                    run_dir = Path(figures_dir) / name / f"UL_{ul}" / f"run_{run_idx:02d}"
                    save_run_figures(run_dir, ul=ul, run_idx=run_idx, results=results)
                    # 甘特图：只画 EO 算法前沿中随机一个解（种子不含 UL，跨 UL 同解）
                    eo_alg = results[ALGORITHMS[0]]
                    gantt_rng = np.random.default_rng([base_seed, inst_idx, run_idx, 2])
                    sol = eo_alg.pareto_front[int(gantt_rng.integers(len(eo_alg.pareto_front)))]
                    save_gantt(sol, inst.processing_times, ul=ul, run_idx=run_idx,
                               rng=gantt_rng, output_dir=run_dir)

            for algo_name in ALGORITHMS:
                st = stats[algo_name]
                fd_mk, fd_tw = f_d[algo_name]
                rows.append({
                    "instance": name,
                    "num_jobs": inst.num_jobs,
                    "num_machines": inst.num_machines,
                    "uncertain_level": ul,
                    "algorithm": algo_name,
                    "f_s_makespan": st["best_mk"],
                    "f_s_twte": st["best_tw"],
                    "f_d_makespan": fd_mk,
                    "f_d_twte": fd_tw,
                    "gap_makespan": (st["best_mk"] - fd_mk) / fd_mk,
                    "gap_twte": (st["best_tw"] - fd_tw) / fd_tw,
                    "nps_mean": round(st["nps"] / num_runs),
                    "cpu_time_mean": st["cpu"] / num_runs,
                })
                if verbose:
                    row = rows[-1]
                    print(f"[{name} | UL={ul} | {algo_name}] "
                          f"f_s=({row['f_s_makespan']:.2f}, {row['f_s_twte']:.2f}), "
                          f"gap=({row['gap_makespan']:.4f}, "
                          f"{row['gap_twte']:.4f}), "
                          f"NPS_mean={row['nps_mean']}, "
                          f"CPU_mean={row['cpu_time_mean']:.3f}s")

            eo, sim_ = ALGORITHMS[0], ALGORITHMS[1]
            p_mk, r_mk = _wilcoxon_signed_rank(samples[eo]["mk"], samples[sim_]["mk"])
            p_tw, r_tw = _wilcoxon_signed_rank(samples[eo]["tw"], samples[sim_]["tw"])
            test_rows.append({"instance": name, "uncertain_level": ul,
                              "p_makespan": p_mk, "r_makespan": r_mk,
                              "p_twte": p_tw, "r_twte": r_tw})

        if verbose:
            print(f"[{name}] 数据集运行完毕，耗时 "
                  f"{(time.perf_counter() - t_inst) / 60.0:.2f} 分钟")

    df = pd.DataFrame(rows, columns=_COLUMNS)
    test_df = pd.DataFrame(test_rows, columns=_TEST_COLUMNS)
    out = Path(output_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        df.to_excel(writer, sheet_name="summary", index=False)
        test_df.to_excel(writer, sheet_name="wilcoxon", index=False)
    if verbose:
        print(f"已导出 {len(df)} 行统计 + {len(test_df)} 行 wilcoxon 检验 → {out}")
    if figures:
        save_gap_boxplots(df, output_dir=figures_dir)
    return df


def launch_experiments(**kwargs) -> pd.DataFrame:
    """正式启动入口：跑完 dataset/ 下所有 xlsx 算例，UL=0.1/0.25/0.5、每档 30 次。

    参数默认值取论文 Table 2（N=100、G=400、cr=0.8、mr=0.2、EO 100 代、短 20 / 长 10000、
    30 次），全部沿用 run_experiments 的默认值；这里只额外固定三件事：全量 dataset、
    UL 序列 (0.1, 0.25, 0.5)、num_runs=30。UL 系用户指定，与论文 §5.1 的 0.5/1.0/2.0 不同。
    数据集与输出路径均锚定到项目根（__file__ 上两级），与当前 cwd 无关；
    传入任意 kwargs 可覆盖（如冒烟测试用小参数、figures=False）。
    """
    root = Path(__file__).resolve().parents[1]            # 项目根（src/experiments.py 上两级）
    dataset_dir = root / "dataset"
    paths = sorted(str(p) for p in dataset_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"未在 {dataset_dir} 找到任何 xlsx 算例")
    params = dict(uncertainty_levels=(0.5, 1, 2),
                  num_runs=30,
                  output_path=str(root / "results" / "experiments.xlsx"),
                  figures_dir=str(root / "results" / "figures"))
    params.update(kwargs)
    return run_experiments(paths, **params)


if __name__ == "__main__":
    launch_experiments()