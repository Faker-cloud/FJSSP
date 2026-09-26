"""绘图原语：单次运行的三张两算法对比图（Pareto / makespan 收敛 / twte 收敛）。

由 src/experiments.py::run_experiments 在每个 (算例, UL, run) 的两算法配对
运行完成后调用——一次演化同时产出统计与图，不重复跑算法。

图片规范：matplotlib Agg 后端、英文标签、PNG 300 dpi、两算法固定配色
（EO 蓝 #2a78d6 / 基线橙 #eb6834，经 dataviz 调色板校验器验证 CVD 安全），
2px 线宽、前沿点标记 ≥6、弱化网格、图例随图（两系列）。
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.algorithms.nsga2.algorithm import SimNSGAII
from src.evaluation import sample_scenario, schedule_trace

ALGO_COLORS = {"EO-Sim-NSGA-II": "#2a78d6", "Sim-NSGA-II": "#eb6834"}

# (目标名, SimNSGAII 上的逐代序列属性, 文件名)
_CONVERGENCE = (("Makespan", "history_makespan", "conv_makespan.png"),
                ("TWTE", "history_twte", "conv_twte.png"))


def _style_axes(ax) -> None:
    """弱化网格与边框（数据为前景，chrome 退后）。"""
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_run_figures(directory, *, ul: float, run_idx: int,
                     results: Dict[str, SimNSGAII]) -> List[Path]:
    """为一个 (算例, UL, run) 的两算法配对结果生成三张图，返回文件路径列表。

    results: {算法名: SimNSGAII 结果对象}，键须在 ALGO_COLORS 中；
    生成于 directory/ 下：pareto.png、conv_makespan.png、conv_twte.png。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # 1. Pareto 前沿对比（横轴 twte，纵轴 makespan）
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo_name, alg in results.items():
        points = sorted((c.twte, c.makespan) for c in alg.pareto_front)
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", markersize=6, linewidth=2,
                color=ALGO_COLORS[algo_name], label=algo_name)
    ax.set_xlabel("TWTE")
    ax.set_ylabel("Makespan")
    ax.set_title(f"Pareto front (UL={ul}, run {run_idx:02d})")
    _style_axes(ax)
    ax.legend(frameon=False)
    out = directory / "pareto.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)

    # 2/3. 收敛曲线（横轴迭代 1..G，纵轴每代选择后种群(N)的最优，两算法同图）
    for objective, attr, fname in _CONVERGENCE:
        fig, ax = plt.subplots(figsize=(7, 5))
        for algo_name, alg in results.items():
            series = getattr(alg, attr)
            ax.plot(range(1, len(series) + 1), series, linewidth=2,
                    color=ALGO_COLORS[algo_name], label=algo_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(objective)
        ax.set_title(f"{objective} per iteration (UL={ul}, run {run_idx:02d})")
        _style_axes(ax)
        ax.legend(frameon=False)
        out = directory / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


_GAP_BOXPLOTS = (("gap_makespan", "Makespan gap", "gap_makespan_boxplot.png"),
                 ("gap_twte", "TWTE gap", "gap_twte_boxplot.png"))
_BOX_OFFSET = 0.18


def save_gap_boxplots(data, output_dir=None) -> List[Path]:
    """用已有 gap 汇总画箱线图（纵轴 gap、横轴 UL，每个 UL 并排 EO/基线两箱）。

    data：tidy DataFrame（含 uncertain_level / algorithm / gap_makespan / gap_twte）
          或已导出的 xlsx 路径；不重跑算法，直接复用 best-of-30 gap。
    分布 = 同一 UL 下所有数据集的 gap 值（每箱 9 点 = 9 个算例）。
    """
    if isinstance(data, (str, Path)):
        data = pd.read_excel(data)
    df = data
    for col in ("uncertain_level", "algorithm", "gap_makespan", "gap_twte"):
        if col not in df.columns:
            raise ValueError(f"gap 箱线图输入缺少列：{col}")

    out_dir = Path(output_dir) if output_dir is not None else Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    algorithms = list(ALGO_COLORS.keys())   # EO 在前（左箱），基线在后（右箱）
    uls = sorted(pd.unique(df["uncertain_level"]))
    paths: List[Path] = []

    for col, ylabel, fname in _GAP_BOXPLOTS:
        fig, ax = plt.subplots(figsize=(7, 5))
        legend_handles = []
        for j, alg in enumerate(algorithms):
            values, positions = [], []
            for i, ul in enumerate(uls):
                sel = df[(df["uncertain_level"] == ul) & (df["algorithm"] == alg)]
                values.append(sel[col].to_numpy())
                positions.append(i + _BOX_OFFSET * (2 * j - 1))
            bp = ax.boxplot(
                values, positions=positions, widths=0.3, patch_artist=True,
                boxprops=dict(facecolor=ALGO_COLORS[alg], alpha=0.55,
                              edgecolor=ALGO_COLORS[alg], linewidth=1.2),
                whiskerprops=dict(color=ALGO_COLORS[alg]),
                capprops=dict(color=ALGO_COLORS[alg]),
                medianprops=dict(color="#0b0b0b", linewidth=2),
                flierprops=dict(marker="o", markerfacecolor=ALGO_COLORS[alg],
                                markersize=5, alpha=0.7,
                                markeredgecolor=ALGO_COLORS[alg]))
            legend_handles.append(bp["boxes"][0])

        ax.set_xticks(list(range(len(uls))))
        ax.set_xticklabels([str(u) for u in uls])
        ax.set_xlabel("Uncertainty level (UL)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by uncertainty level")
        ax.legend(legend_handles, algorithms, frameon=False)
        _style_axes(ax)
        out = out_dir / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def save_gantt(chromosome, processing_times, *, ul: float, run_idx: int,
               rng, output_dir) -> Path:
    """画单个解的甘特图（横轴 = 时间，纵轴 = 机器，对齐论文 Fig.3）。

    加工时间用该 UL 下采样一次的对数正态场景；每机器一条泳道，每道操作一条横向条，
    颜色按作业（tab20 循环）、条内标注操作序号 O_k（k = 该作业第 k 道操作）；
    右侧图例列出「作业 → 颜色」（作业为颜色主标识，颜色仅辅助区分）。
    """
    scenario = sample_scenario(processing_times, ul, rng)
    trace = schedule_trace(chromosome, scenario)   # (N,5): job, op, machine, start, finish

    job_colors = plt.cm.tab20.colors
    num_machines = chromosome.num_machines
    num_jobs = chromosome.num_jobs

    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * num_machines + 1)))
    for job, op, machine, start, finish in trace:
        j = int(job)
        y = int(machine) - 1
        ax.barh(y, finish - start, left=start, height=0.7,
                color=job_colors[(j - 1) % len(job_colors)],
                edgecolor="white", linewidth=0.5)
        ax.text(start + (finish - start) / 2.0, y, f"O{int(op)}",
                ha="center", va="center", fontsize=8, color="#0b0b0b")

    ax.set_yticks(list(range(num_machines)))
    ax.set_yticklabels([f"M{i + 1}" for i in range(num_machines)])
    ax.set_xlim(left=0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"Gantt (UL={ul}, run {run_idx:02d})")
    _style_axes(ax)

    # 右侧图例：作业 → 颜色（多作业时分列）
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=job_colors[(j - 1) % len(job_colors)])
               for j in range(1, num_jobs + 1)]
    leg = ax.legend(handles, [f"Job {j}" for j in range(1, num_jobs + 1)],
                    loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                    fontsize=8, ncol=max(1, (num_jobs + 14) // 15))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "gantt.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    return out


if __name__ == "__main__":
    src = Path("results/experiments.xlsx")
    if src.exists():
        for f in save_gap_boxplots(src):
            print("已生成：", f)
    else:
        print(f"未找到 {src}，请先运行 run_experiments 生成结果表")