"""实验层：批量运行 + 论文 §5.1 口径统计 + 画图 + xlsx 导出。

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
wilcoxon（每行 = 算例 × UL，含两目标的 p 值与 R）。每个 run 生成四张图
（Pareto / makespan 收敛 / twte 收敛 / 甘特图，见 src/plotting.py），并在全部算例跑完后
按同一份 df 另画两张跨算例 gap 箱线图，与统计共用同一次演化，不重复计算。

参数默认值取自论文 Table 2（N=100、G=400、cr=0.8、mr=0.2、EO 100 代、短 20 / 长 10000）；
EO 自由参数 a1/a2/gp 默认 2.0/1.0/0.5（EO 论文 3.1.3/3.1.4），均可覆盖。

随机版种子为**配对方案**（公共随机数）：每个 (算例, 运行) 内两算法的 NSGA-II 演化用
同一种子流 `[base_seed, 算例, 运行, 0]`，EO 阶段独立流 `[base_seed, 算例, 运行, 1]`，
基线随机初始化独立流 `[base_seed, 算例, 运行, 3]`（不消耗共享流）——两算法从初始种群
评估起，MC 场景、锦标赛、交叉变异的抽样序列逐位对齐（严格公共随机数，配对比较可消去
场景抽样噪声）；不同运行种子不同，且**种子不含 UL**——同一数据集下不同 UL 水平的同一
run 复用同一种子。

num_workers > 1 时以多进程并行执行：任务粒度 = 一个 (算例, UL, run) 的两算法配对运行
（公共随机数要求两算法在同一任务内先后执行）+ 每算例 2 个确定性 f_d 任务；每次
_run_once 都用种子新建独立随机流、任务之间无共享状态，故**并行结果与串行逐位一致**
（cpu_time_mean 含计时噪声除外）。并行下「数据集耗时」= 该算例全部 run 任务的首尾时间
跨度（各算例并行重叠，不再等于总耗时）。

正式启动入口为 `launch_experiments()`；`run_experiments` 仍可独立调用以指定任意算例子集
或覆盖参数。
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.algorithms.eo.optimizer import EquilibriumOptimizer
from src.algorithms.nsga2.algorithm import SimNSGAII
from src.data_structures.chromosome import FJSSPChromosome
from src.data_structures.instance import Instance
from src.dataset import load_instance
from src.plotting import save_run_figures, save_gap_boxplots, save_gantt

ALGORITHMS = ("EO-Sim-NSGA-II", "Sim-NSGA-II")   # 下标 0 = EO 版，1 = 基线

_COLUMNS = ["instance", "num_jobs", "num_machines", "uncertain_level", "algorithm",
            "f_s_makespan", "f_s_twte", "f_d_makespan", "f_d_twte",
            "gap_makespan", "gap_twte", "nps_mean", "cpu_time_mean"]

_TEST_COLUMNS = ["instance", "uncertain_level",
                 "p_makespan", "r_makespan", "p_twte", "r_twte"]

# 随任务下发的演化参数（键与 _run_once 的同名形参一致，Table 2 口径 + EO 自由参数）
_EVOLVE_KEYS = ("population_size", "num_generations", "crossover_rate",
                "mutation_rate", "eo_max_iter", "s_short", "s_long",
                "a1", "a2", "gp")


def _build_initial_population(instance: Instance, *, use_eo: bool, population_size: int,
                              eo_max_iter: int, a1: float, a2: float, gp: float,
                              rng: np.random.Generator,
                              eo_rng: Optional[np.random.Generator] = None,
                              init_rng: Optional[np.random.Generator] = None
                              ) -> List[FJSSPChromosome]:
    """构建 NSGA-II 初始种群：use_eo=True 时 EO 整群解码，否则随机生成。

    EO 阶段用 eo_rng（缺省退 rng）；随机初始化用 init_rng（缺省退 rng）——初始化的
    随机流与 NSGA-II 演化流分离，两算法进入演化时共享流位置一致（公共随机数）。
    """
    if use_eo:
        eo = EquilibriumOptimizer(instance.num_jobs, instance.num_machines,
                                  instance.processing_times,
                                  population_size=population_size,
                                  max_iter=eo_max_iter,
                                  a1=a1, a2=a2, GP=gp,
                                  rng=rng if eo_rng is None else eo_rng).optimize()
        return eo.chromosomes()
    rng_init = init_rng if init_rng is not None else rng
    return [FJSSPChromosome.random(instance.num_jobs, instance.num_machines, rng_init)
            for _ in range(population_size)]


def _run_once(instance: Instance, *, use_eo: bool, sim: bool, uncertain_level: float,
              population_size: int, num_generations: int, crossover_rate: float,
              mutation_rate: float, eo_max_iter: int, s_short: int, s_long: int,
              a1: float = 2.0, a2: float = 1.0, gp: float = 0.5,
              rng: np.random.Generator,
              eo_rng: Optional[np.random.Generator] = None,
              init_rng: Optional[np.random.Generator] = None):
    """单次完整演化：构建初始种群（EO / 随机，不计时）+ NSGA-II（计时）。

    返回 (SimNSGAII 结果对象, NSGA-II 阶段耗时秒)。CPU Time 口径仅含 NSGA-II 演化。
    a1 / a2 / gp 为 EO 自由参数（论文 3.1.3/3.1.4 默认 2.0 / 1.0 / 0.5，即 GP）。
    """
    initial_population = _build_initial_population(
        instance, use_eo=use_eo, population_size=population_size,
        eo_max_iter=eo_max_iter, a1=a1, a2=a2, gp=gp,
        rng=rng, eo_rng=eo_rng, init_rng=init_rng)

    t0 = time.perf_counter()
    alg = SimNSGAII(instance.num_jobs, instance.num_machines, instance.processing_times,
                    instance.priorities, instance.due_dates,
                    population_size=population_size,
                    num_generations=num_generations,
                    crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                    uncertain_level=uncertain_level, sim=sim,
                    s_short=s_short, s_long=s_long,
                    rng=rng).optimize(initial_population)
    return alg, time.perf_counter() - t0


def _pareto_extremes(alg: SimNSGAII):
    """长模拟重评价后前沿上两目标各自的最优值（前沿含种群各目标极值点）。"""
    return (min(c.makespan for c in alg.pareto_front),
            min(c.twte for c in alg.pareto_front))


def _wilcoxon_signed_rank(eo, baseline, alpha: float = 0.05):
    """配对 Wilcoxon 符号秩检验（精确法，双尾），返回 (p_value, R)。

    R ∈ {'+', '−', '='}：p < alpha 时按中位数方向判定（两目标均最小化，中位数小者更优）
    EO 更优记 '+'、基线更优记 '−'；p ≥ alpha 记 '='。精确法不处理零差，先丢弃；
    全零差返回 (1.0, '=')。
    """
    x = np.asarray(eo, dtype=float)
    y = np.asarray(baseline, dtype=float)
    keep = (x - y) != 0.0
    xf, yf = x[keep], y[keep]
    if xf.size == 0:
        return 1.0, "="
    _, p = wilcoxon(xf, yf, method="exact")
    if p >= alpha:
        return p, "="
    return p, ("+" if np.median(x) < np.median(y) else "-")


# --------------------------------------------------------------------------- #
# 任务函数（模块顶层，供多进程 pickle 引用；串行与并行共用，保证行为一致）
# --------------------------------------------------------------------------- #

def _run_deterministic(args) -> dict:
    """任务：一个算例的一次确定性 f_d 运行（algo_idx=0 EO 版 / 1 基线）。

    种子 [base_seed, inst_idx, algo_idx] 只依赖索引，与执行进程/顺序无关。
    返回 {inst_idx, algo_idx, makespan, twte}。
    """
    path, inst_idx, algo_idx, base_seed, params = args
    instance = load_instance(path)
    rng = np.random.default_rng([base_seed, inst_idx, algo_idx])
    alg_det, _ = _run_once(
        instance, use_eo=(algo_idx == 0), sim=False, uncertain_level=0.0,
        **{k: params[k] for k in _EVOLVE_KEYS}, rng=rng)
    makespan, twte = _pareto_extremes(alg_det)
    return {"inst_idx": inst_idx, "algo_idx": algo_idx,
            "makespan": makespan, "twte": twte}


def _execute_paired_run(args) -> dict:
    """任务：一个 (算例, UL, run) 的两算法配对运行（公共随机数）+ 该 run 的四张图。

    串行与并行共用此函数，保证两种模式行为一致：种子只依赖
    (base_seed, inst_idx, run_idx) 且不含 UL（跨 UL 同 run 复用同一种子）；
    基线随机初始化走独立子流（不消耗共享流），故两算法的 NSGA-II 演化流从头对齐
    （严格公共随机数）。图在执行进程内生成（目录互不重叠）。
    返回逐算法的 {makespan, twte, nps, cpu} 与任务起止时间戳（time.time()，跨进程可比）。
    """
    (path, inst_idx, ul_index, uncertain_level, run_idx, num_runs, base_seed,
     figures_dir, verbose, params) = args
    instance = load_instance(path)
    name = Path(path).stem
    evolve = {k: params[k] for k in _EVOLVE_KEYS}
    t_start = time.time()
    nsga_seed = [base_seed, inst_idx, run_idx, 0]   # 两算法共用的 NSGA-II 种子流
    eo_seed = [base_seed, inst_idx, run_idx, 1]     # EO 阶段独立子流
    init_seed = [base_seed, inst_idx, run_idx, 3]   # 基线随机初始化独立子流
    results, algos = {}, {}
    for algo_idx, algo_name in enumerate(ALGORITHMS):
        if verbose:
            print(f"[{name} | UL={uncertain_level}] "
                  f"run {run_idx + 1}/{num_runs}: {algo_name}")
        alg, cpu = _run_once(
            instance, use_eo=(algo_idx == 0), sim=True,
            uncertain_level=uncertain_level, **evolve,
            rng=np.random.default_rng(nsga_seed),
            eo_rng=(np.random.default_rng(eo_seed) if algo_idx == 0 else None),
            init_rng=(None if algo_idx == 0
                      else np.random.default_rng(init_seed)))
        results[algo_name] = alg
        makespan, twte = _pareto_extremes(alg)
        algos[algo_name] = {"makespan": makespan, "twte": twte,
                            "nps": len(alg.pareto_front), "cpu": cpu}
    _save_run_artifacts(results, instance, name, uncertain_level, run_idx,
                        base_seed, inst_idx, figures_dir)
    return {"inst_idx": inst_idx, "ul_index": ul_index, "run_idx": run_idx,
            "t_start": t_start, "t_end": time.time(), "algos": algos}


def _save_run_artifacts(results, instance, name, uncertain_level, run_idx,
                        base_seed, inst_idx, figures_dir) -> None:
    """生成一个 run 的四张图（Pareto / 收敛 ×2 / 甘特图）。"""
    run_dir = Path(figures_dir) / name / f"UL_{uncertain_level}" / f"run_{run_idx:02d}"
    save_run_figures(run_dir, ul=uncertain_level, run_idx=run_idx, results=results)
    # 甘特图：只画 EO 算法前沿中随机一个解（种子不含 UL，跨 UL 同解）
    eo_result = results[ALGORITHMS[0]]
    gantt_rng = np.random.default_rng([base_seed, inst_idx, run_idx, 2])
    sol = eo_result.pareto_front[int(gantt_rng.integers(len(eo_result.pareto_front)))]
    save_gantt(sol, instance.processing_times, ul=uncertain_level, run_idx=run_idx,
               rng=gantt_rng, output_dir=run_dir)


def _summarize_uncertainty_level(name: str, instance: Instance,
                                 uncertain_level: float, num_runs: int,
                                 f_d: Dict[str, Tuple[float, float]],
                                 payloads: Sequence[dict]):
    """把一个 (算例, UL) 的 num_runs 个配对结果（按 run 序）聚成 2 行 summary + 1 行 wilcoxon。"""
    stats = {algo: {"best_makespan": np.inf, "best_twte": np.inf,
                    "nps": 0, "cpu": 0.0} for algo in ALGORITHMS}
    samples = {algo: {"makespan": [], "twte": []} for algo in ALGORITHMS}
    for payload in payloads:
        for algo_name in ALGORITHMS:
            a = payload["algos"][algo_name]
            samples[algo_name]["makespan"].append(a["makespan"])
            samples[algo_name]["twte"].append(a["twte"])
            st = stats[algo_name]
            st["best_makespan"] = min(st["best_makespan"], a["makespan"])
            st["best_twte"] = min(st["best_twte"], a["twte"])
            st["nps"] += a["nps"]
            st["cpu"] += a["cpu"]
    summary_rows = []
    for algo_name in ALGORITHMS:
        st = stats[algo_name]
        deterministic_makespan, deterministic_twte = f_d[algo_name]
        summary_rows.append({
            "instance": name,
            "num_jobs": instance.num_jobs,
            "num_machines": instance.num_machines,
            "uncertain_level": uncertain_level,
            "algorithm": algo_name,
            "f_s_makespan": st["best_makespan"],
            "f_s_twte": st["best_twte"],
            "f_d_makespan": deterministic_makespan,
            "f_d_twte": deterministic_twte,
            "gap_makespan": (st["best_makespan"] - deterministic_makespan) / deterministic_makespan,
            "gap_twte": (st["best_twte"] - deterministic_twte) / deterministic_twte,
            "nps_mean": round(st["nps"] / num_runs),
            "cpu_time_mean": st["cpu"] / num_runs,
        })
    eo, baseline = ALGORITHMS[0], ALGORITHMS[1]
    p_makespan, r_makespan = _wilcoxon_signed_rank(
        samples[eo]["makespan"], samples[baseline]["makespan"])
    p_twte, r_twte = _wilcoxon_signed_rank(
        samples[eo]["twte"], samples[baseline]["twte"])
    test_row = {"instance": name, "uncertain_level": uncertain_level,
                "p_makespan": p_makespan, "r_makespan": r_makespan,
                "p_twte": p_twte, "r_twte": r_twte}
    return summary_rows, test_row


def _print_summary_rows(name: str, uncertain_level: float,
                        summary_rows: List[dict], verbose: bool) -> None:
    """打印一个 (算例, UL) 的两条 per-row 汇总（与原串行输出格式一致）。"""
    if not verbose:
        return
    for row in summary_rows:
        print(f"[{name} | UL={uncertain_level} | {row['algorithm']}] "
              f"f_s=({row['f_s_makespan']:.2f}, {row['f_s_twte']:.2f}), "
              f"gap=({row['gap_makespan']:.4f}, {row['gap_twte']:.4f}), "
              f"NPS_mean={row['nps_mean']}, "
              f"CPU_mean={row['cpu_time_mean']:.3f}s")


def _export_results(df: pd.DataFrame, test_df: pd.DataFrame, output_path: str) -> Path:
    """把 summary 与 wilcoxon 两个 sheet 写入 xlsx，返回输出路径。"""
    out = Path(output_path)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as writer:
        df.to_excel(writer, sheet_name="summary", index=False)
        test_df.to_excel(writer, sheet_name="wilcoxon", index=False)
    return out


def _save_gap_boxplots(df: pd.DataFrame, figures_dir: str) -> None:
    """按汇总表另画两张跨算例 gap 箱线图。"""
    save_gap_boxplots(df, output_dir=figures_dir)


def run_experiments(instance_paths: Sequence[str],
                    population_size: int = 100, num_generations: int = 400,
                    crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                    eo_max_iter: int = 100, s_short: int = 20, s_long: int = 10000,
                    a1: float = 2.0, a2: float = 1.0, gp: float = 0.5,
                    uncertainty_levels: Sequence[float] = (0.5, 1.0, 2.0),
                    num_runs: int = 30, base_seed: int = 0,
                    output_path: str = "results/experiments.xlsx",
                    figures_dir: str = "results/figures",
                    verbose: bool = True,
                    num_workers: Optional[int] = None) -> pd.DataFrame:
    """批量实验入口：逐算例统计 f_s / f_d / gap / NPS / CPU，画图并导出 xlsx，返回汇总表。

    a1 / a2 / gp 为 EO 自由参数（论文 3.1.3/3.1.4 默认 2.0 / 1.0 / 0.5，
    gp 对应 EquilibriumOptimizer 的 GP），仅影响 EO 阶段。

    每个 (算例, UL, run) 生成四张图到 {figures_dir}/{算例}/UL_{ul}/run_{r:02d}/；
    全部跑完后在 {figures_dir} 根再画两张 gap 箱线图。

    num_workers：并行 worker 进程数。None → 全部逻辑核心（os.cpu_count()）；
    1 → 串行。并行结果与串行逐位一致（cpu_time_mean 含计时噪声除外），
    详见模块 docstring 的并行说明。
    """
    t_total = time.perf_counter()
    workers = (os.cpu_count() or 1) if num_workers is None else num_workers
    params = {"population_size": population_size, "num_generations": num_generations,
              "crossover_rate": crossover_rate, "mutation_rate": mutation_rate,
              "eo_max_iter": eo_max_iter, "s_short": s_short, "s_long": s_long,
              "a1": a1, "a2": a2, "gp": gp}

    instances = [load_instance(p) for p in instance_paths]
    names = [Path(p).stem for p in instance_paths]
    num_instances = len(instance_paths)
    num_uncertainty_levels = len(uncertainty_levels)

    rows: List[dict] = []
    test_rows: List[dict] = []

    if workers <= 1:
        # ---- 串行：与原实现完全同序（每算例 f_d → 逐 UL 逐 run → 聚合） ----
        for inst_idx in range(num_instances):
            t_inst = time.perf_counter()   # 计时起点：覆盖本算例 f_d + 全部 UL×run
            f_d = {}
            for algo_idx in range(len(ALGORITHMS)):
                r = _run_deterministic((instance_paths[inst_idx], inst_idx, algo_idx,
                                        base_seed, params))
                f_d[ALGORITHMS[r["algo_idx"]]] = (r["makespan"], r["twte"])
            for ul_index, uncertain_level in enumerate(uncertainty_levels):
                payloads = [_execute_paired_run(
                    (instance_paths[inst_idx], inst_idx, ul_index, uncertain_level,
                     run_idx, num_runs, base_seed, figures_dir, verbose, params))
                    for run_idx in range(num_runs)]
                summary_rows, test_row = _summarize_uncertainty_level(
                    names[inst_idx], instances[inst_idx], uncertain_level,
                    num_runs, f_d, payloads)
                rows.extend(summary_rows)
                test_rows.append(test_row)
                _print_summary_rows(names[inst_idx], uncertain_level,
                                    summary_rows, verbose)
            if verbose:
                print(f"[{names[inst_idx]}] 数据集运行完毕，耗时 "
                      f"{(time.perf_counter() - t_inst) / 60.0:.2f} 分钟")
    else:
        # ---- 并行：任务互相独立（种子只依赖索引），结果与串行逐位一致 ----
        summaries: Dict[Tuple[int, int], Tuple[List[dict], dict]] = {}
        run_results: Dict[Tuple[int, int], Dict[int, dict]] = {}
        deterministic_results: Dict[Tuple[int, int], Tuple[float, float]] = {}
        runs_pending = {(i, u): num_runs for i in range(num_instances)
                        for u in range(num_uncertainty_levels)}
        instance_remaining = {i: 2 + num_uncertainty_levels * num_runs
                              for i in range(num_instances)}
        instance_span = {i: [float("inf"), 0.0] for i in range(num_instances)}  # [min_start, max_end]

        def summarize_when_ready(inst_idx: int, ul_index: int) -> None:
            """某 (算例, UL) 的 run 全部到齐且 f_d 就绪 → 聚合 + 打印（完成序）。"""
            key = (inst_idx, ul_index)
            if (key in summaries or runs_pending[key] > 0
                    or (inst_idx, 0) not in deterministic_results
                    or (inst_idx, 1) not in deterministic_results):
                return
            f_d = {ALGORITHMS[a]: deterministic_results[(inst_idx, a)]
                   for a in range(len(ALGORITHMS))}
            payloads = [run_results[key][r] for r in range(num_runs)]
            summary_rows, test_row = _summarize_uncertainty_level(
                names[inst_idx], instances[inst_idx],
                uncertainty_levels[ul_index], num_runs, f_d, payloads)
            summaries[key] = (summary_rows, test_row)
            _print_summary_rows(names[inst_idx], uncertainty_levels[ul_index],
                                summary_rows, verbose)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for inst_idx in range(num_instances):
                path = instance_paths[inst_idx]
                for algo_idx in range(len(ALGORITHMS)):
                    fut = pool.submit(_run_deterministic, (path, inst_idx, algo_idx,
                                                           base_seed, params))
                    futures[fut] = inst_idx
                for ul_index, uncertain_level in enumerate(uncertainty_levels):
                    for run_idx in range(num_runs):
                        fut = pool.submit(
                            _execute_paired_run,
                            (path, inst_idx, ul_index, uncertain_level, run_idx,
                             num_runs, base_seed, figures_dir, verbose, params))
                        futures[fut] = inst_idx

            for fut in as_completed(futures):
                inst_idx = futures[fut]
                r = fut.result()
                if "algo_idx" in r:                       # 确定性 f_d 任务
                    deterministic_results[(inst_idx, r["algo_idx"])] = \
                        (r["makespan"], r["twte"])
                    for u in range(num_uncertainty_levels):
                        summarize_when_ready(inst_idx, u)
                else:                                     # 配对运行任务
                    key = (inst_idx, r["ul_index"])
                    run_results.setdefault(key, {})[r["run_idx"]] = r
                    runs_pending[key] -= 1
                    span = instance_span[inst_idx]
                    span[0] = min(span[0], r["t_start"])
                    span[1] = max(span[1], r["t_end"])
                    summarize_when_ready(inst_idx, r["ul_index"])
                instance_remaining[inst_idx] -= 1
                if instance_remaining[inst_idx] == 0:
                    for u in range(num_uncertainty_levels):   # 兜底（f_d 晚到的情况）
                        summarize_when_ready(inst_idx, u)
                    if verbose:
                        print(f"[{names[inst_idx]}] 数据集运行完毕，耗时 "
                              f"{(instance_span[inst_idx][1] - instance_span[inst_idx][0]) / 60.0:.2f} 分钟")

        # 按串行同序组装（完成序只影响打印，不影响导出顺序）
        for inst_idx in range(num_instances):
            for ul_index in range(num_uncertainty_levels):
                summary_rows, test_row = summaries[(inst_idx, ul_index)]
                rows.extend(summary_rows)
                test_rows.append(test_row)

    df = pd.DataFrame(rows, columns=_COLUMNS)
    test_df = pd.DataFrame(test_rows, columns=_TEST_COLUMNS)
    out = _export_results(df, test_df, output_path)
    if verbose:
        print(f"已导出 {len(df)} 行统计 + {len(test_df)} 行 wilcoxon 检验 → {out}")
        print(f"实验总耗时 {(time.perf_counter() - t_total) / 60.0:.2f} 分钟"
              f"（{workers} 进程{'并行' if workers > 1 else '串行'}）")
    _save_gap_boxplots(df, figures_dir)
    return df


def launch_experiments(**kwargs) -> pd.DataFrame:
    """正式启动入口：跑完 dataset/ 下所有 xlsx 算例。

    参数默认值沿用 run_experiments（论文 Table 2 口径），这里额外固定：全量 dataset、
    UL 序列 (0.5, 1.0, 2.0)、num_runs=10、num_workers=8、EO 参数 a1=2.1 / a2=1.2。
    数据集与输出路径均锚定到项目根（__file__ 上两级），与当前 cwd 无关；
    传入任意 kwargs 可覆盖（如冒烟测试用小参数）。
    """
    root = Path(__file__).resolve().parents[1]            # 项目根（src/experiments.py 上两级）
    dataset_dir = root / "dataset"
    paths = sorted(str(p) for p in dataset_dir.glob("*.xlsx"))
    if not paths:
        raise FileNotFoundError(f"未在 {dataset_dir} 找到任何 xlsx 算例")
    params = dict(uncertainty_levels=(0.5, 1, 2),
                  num_runs=10,
                  num_workers=8,
                  a1=2.1,
                  a2=1.2,
                  output_path=str(root / "results" / "experiments.xlsx"),
                  figures_dir=str(root / "results" / "figures"))
    params.update(kwargs)
    return run_experiments(paths, **params)


if __name__ == "__main__":
    # TODO 1. 选择父代时，优先考虑twte小的作为父代（因为使用EO之后makespan会降低）
    # TODO 2. 在EO的memory saving中添加概率，低于这个概率就可以保留较差的解，避免陷入局部最优
    # TODO 3. EO可能需要早停
    launch_experiments()