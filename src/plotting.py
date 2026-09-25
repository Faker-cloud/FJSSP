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

from src.nsga2.algorithm import SimNSGAII

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

    # 2/3. 收敛曲线（横轴迭代 1..G，纵轴每代合并种群最优，两算法同图）
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


if __name__ == "__main__":
    print()