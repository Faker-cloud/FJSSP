"""NSGA-II 变异算子：OS 随机两位交换 + MS 单作业块内互换。

流程（用户确定，复合变异）：
- 每个子代以概率 mr 整体触发一次变异；触发时同时做两步：
    1. OS 变异——随机两位交换：随机选两个不同位置，交换二者的作业号。
    2. MS 变异——单作业块内互换：随机抽一个作业，在其 M 台机器段内
       随机选两个不同位置交换。

两条变异都保持子代合法：OS 交换保持作业多重集不变（每作业仍出现 M 次）；
MS 块内互换仍为 1..M 的排列。

示例（用户给定，J=3, M=3）：
    OS 抽到第 5、6 位（值 1 与 2）交换；
    MS 抽到 J3 段的前两位交换，(1,3,2) → (3,1,2)。
"""

from typing import Optional

import numpy as np

from src.data_structures.chromosome import FJSSPChromosome


def os_swap_mutation(os: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """OS 变异：随机选两个不同位置交换作业号，返回新 OS 数组。"""
    os = np.asarray(os, dtype=int).copy()
    if os.size < 2:
        return os
    i, j = rng.choice(os.size, size=2, replace=False)
    os[i], os[j] = os[j], os[i]
    return os


def ms_block_swap_mutation(ms: np.ndarray, num_jobs: int, num_machines: int,
                           rng: np.random.Generator) -> np.ndarray:
    """MS 变异：随机抽一个作业，在其机器段内随机选两个不同位置交换，返回新 MS 数组。"""
    ms = np.asarray(ms, dtype=int).copy()
    if num_machines < 2:
        return ms
    job = int(rng.integers(num_jobs))
    a, b = rng.choice(num_machines, size=2, replace=False)
    base = job * num_machines
    ms[base + a], ms[base + b] = ms[base + b], ms[base + a]
    return ms


def mutate(chromosome: FJSSPChromosome, mr: float,
           rng: Optional[np.random.Generator] = None) -> FJSSPChromosome:
    """复合变异：以概率 mr 同时做 OS 两位交换与 MS 单作业块内互换，返回变异后的解。

    未触发（rng.random() >= mr）时返回原解副本；mr=1 时必定变异。
    """
    if not 0.0 <= mr <= 1.0:
        raise ValueError(f"变异率 mr 应在 [0,1] 内，实际为 {mr}")
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() >= mr:
        return chromosome.copy()

    os_new = os_swap_mutation(chromosome.os, rng)
    ms_new = ms_block_swap_mutation(
        chromosome.ms, chromosome.num_jobs, chromosome.num_machines, rng)
    return FJSSPChromosome(os_new, ms_new, chromosome.num_machines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 用户示例用染色体（复用交叉示例的 P1）
    chromo = FJSSPChromosome(
        os=[1, 2, 2, 3, 1, 3, 2, 1, 3],
        ms=[3, 1, 2, 1, 3, 2, 2, 1, 3], num_machines=3)

    # OS 变异：多重集不变，且至多两处不同（0 或 2 处）
    os_new = os_swap_mutation(chromo.os, rng)
    same_multiset = np.array_equal(np.sort(os_new), np.sort(chromo.os))
    n_diff_os = int(np.sum(os_new != chromo.os))
    print("OS 变异:", chromo.os.tolist(), "->", os_new.tolist(),
          f"(多重集不变={same_multiset}, 不同位置数={n_diff_os})")
    assert same_multiset and n_diff_os in (0, 2)

    # MS 变异：每作业段仍是 1..M 排列，且至多一个作业段被两位置交换
    ms_new = ms_block_swap_mutation(chromo.ms, chromo.num_jobs,
                                    chromo.num_machines, rng)
    blocks_old = chromo.ms.reshape(3, 3)
    blocks_new = ms_new.reshape(3, 3)
    expected = np.tile(np.arange(1, 4), (3, 1))
    assert np.array_equal(np.sort(blocks_new, axis=1), expected)
    row_diff = np.sum(np.any(blocks_old != blocks_new, axis=1))
    print("MS 变异:", chromo.ms.tolist(), "->", ms_new.tolist(),
          f"(每段仍为排列=True, 变化的作业段数={row_diff})")
    assert row_diff <= 1

    # 复合变异：mr=1 必变异且合法；mr=0 不变
    m1 = mutate(chromo, 1.0, rng=np.random.default_rng(3))
    m0 = mutate(chromo, 0.0, rng=np.random.default_rng(4))
    print("\nmutate(mr=1.0):", m1)
    print("合法性 =", m1.is_valid())
    assert m1.is_valid()
    assert np.array_equal(m0.os, chromo.os) and np.array_equal(m0.ms, chromo.ms)

    # mr=0.5 多轮统计：约一半触发，全部合法
    rng2 = np.random.default_rng(5)
    n_changed, n_trials = 0, 2000
    for _ in range(n_trials):
        c = mutate(chromo, 0.5, rng=rng2)
        assert c.is_valid()
        if not (np.array_equal(c.os, chromo.os) and np.array_equal(c.ms, chromo.ms)):
            n_changed += 1
    print(f"mutate(mr=0.5) 多轮：触发比例 = {n_changed / n_trials:.3f}（预期约 0.5），全部合法 = True")