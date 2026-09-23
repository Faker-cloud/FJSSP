"""NSGA-II 交叉算子：OS 层 POX + MS 层段粒度交叉。

流程（用户确定）：
- OS 层 POX（Precedence Operation Crossover）：
    1. 随机作业子集 JA（其余作业为 JB）。
    2. 把父代 P1 中作业号 ∈JA 的位置（含位置本身）照抄到子代 1。
    3. 空位按 P2 去掉 JA 后得到的 JB 作业子序列顺序依次填入。
  第二个子代对称交换继承方向（JA 位置取 P2、空位按 P1 的 JB 顺序填）。
- MS 层段粒度交叉：
    每个作业独立掷硬币选父代，整段搬运该作业的 M 台机器号；
    第二个子代取与子代 1 相反的父代段（互补）。
- 每次交叉产 2 个子代。

两条规则都保持子代合法：POX 保持作业多重集不变（每作业仍出现 M 次）；
MS 整段搬运的是合法父代中 1..M 的机器排列。

示例（用户给定 P1/P2, J=3, M=3，JA={J1}，J1←P2、J2←P1、J3←P2）：
    P1 OS=(1,2,2,3,1,3,2,1,3)  MS=(3,1,2 | 1,3,2 | 2,1,3)
    P2 OS=(2,3,1,1,3,2,3,1,2)  MS=(2,3,1 | 3,2,1 | 1,3,2)
    child1 OS=(1,2,3,3,1,2,3,1,2)  MS=(2,3,1 | 1,3,2 | 1,3,2)
    child2 OS=(2,2,1,1,3,3,2,1,3)  MS=(3,1,2 | 3,2,1 | 2,1,3)
"""

from typing import Tuple

import numpy as np

from src.problem.fjssp_chromosome import FJSSPChromosome


def _sample_job_subset(num_jobs: int, rng: np.random.Generator) -> np.ndarray:
    """随机作业子集 JA：每作业独立 50/50 进 JA；若为空集或全集则重采样，
    避免交叉退化为只复制父代。返回 (J,) bool 掩码，True 表示该作业 ∈JA。"""
    while True:
        ja_mask = rng.random(num_jobs) < 0.5
        if ja_mask.any() and not ja_mask.all():
            return ja_mask


def pox_os_crossover(os1: np.ndarray, os2: np.ndarray,
                     ja_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OS 层 POX，返回 (child1_os, child2_os)。

    ja_mask: (J,) bool，True 表示该作业属于 JA。
    child1：JA 位置照抄 os1（含位置本身），空位按 os2 的 JB 作业顺序填入；
    child2：对称交换继承方向（JA 位置照抄 os2，空位按 os1 的 JB 顺序填）。
    """
    os1 = np.asarray(os1, dtype=int)
    os2 = np.asarray(os2, dtype=int)

    # child1
    fix1 = ja_mask[os1 - 1]                 # os1 中 ∈JA 的位置
    child1 = np.empty_like(os1)
    child1[fix1] = os1[fix1]                # JA 位置照抄 P1
    child1[~fix1] = os2[~ja_mask[os2 - 1]]  # 空位按 P2 的 JB 顺序填

    # child2（对称）
    fix2 = ja_mask[os2 - 1]
    child2 = np.empty_like(os2)
    child2[fix2] = os2[fix2]
    child2[~fix2] = os1[~ja_mask[os1 - 1]]

    return child1, child2


def segment_ms_crossover(ms1: np.ndarray, ms2: np.ndarray,
                         num_jobs: int, num_machines: int,
                         take2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MS 层段粒度交叉，返回 (child1_ms, child2_ms)。

    take2: (J,) bool，True 表示 child1 该作业取 P2 的机器段，False 取 P1；
    child2 取 child1 的互补侧（即 take2 处取 P1、其余取 P2）。
    """
    ms1 = np.asarray(ms1, dtype=int).reshape(num_jobs, num_machines)
    ms2 = np.asarray(ms2, dtype=int).reshape(num_jobs, num_machines)
    child1 = np.where(take2[:, None], ms2, ms1).ravel()
    child2 = np.where(take2[:, None], ms1, ms2).ravel()
    return child1, child2


def crossover(parent1: FJSSPChromosome, parent2: FJSSPChromosome,
              rng: np.random.Generator) -> Tuple[FJSSPChromosome, FJSSPChromosome]:
    """对两个父代做 OS 层 POX + MS 层段粒度交叉，返回两个子代。"""
    if parent1.num_jobs != parent2.num_jobs or parent1.num_machines != parent2.num_machines:
        raise ValueError("两个父代的作业数/机器数必须一致")
    num_jobs = parent1.num_jobs
    num_machines = parent1.num_machines

    ja_mask = _sample_job_subset(num_jobs, rng)
    take2 = rng.random(num_jobs) < 0.5          # True → child1 该作业取 P2 段

    child1_os, child2_os = pox_os_crossover(parent1.os, parent2.os, ja_mask)
    child1_ms, child2_ms = segment_ms_crossover(
        parent1.ms, parent2.ms, num_jobs, num_machines, take2)

    return (FJSSPChromosome(child1_os, child1_ms, num_machines),
            FJSSPChromosome(child2_os, child2_ms, num_machines))


if __name__ == "__main__":
    # 用户示例（J=3, M=3）：固定 JA={J1}、固定 MS 取侧，复现手推结果
    P1 = FJSSPChromosome(
        os=[1, 2, 2, 3, 1, 3, 2, 1, 3],
        ms=[3, 1, 2, 1, 3, 2, 2, 1, 3], num_machines=3)
    P2 = FJSSPChromosome(
        os=[2, 3, 1, 1, 3, 2, 3, 1, 2],
        ms=[2, 3, 1, 3, 2, 1, 1, 3, 2], num_machines=3)

    ja_mask = np.array([True, False, False])    # JA = {J1}
    os1, os2 = pox_os_crossover(P1.os, P2.os, ja_mask)
    take2 = np.array([True, False, True])       # J1←P2, J2←P1, J3←P2
    ms1, ms2 = segment_ms_crossover(P1.ms, P2.ms, 3, 3, take2)

    print("OS POX (JA={J1}):")
    print("  child1 OS =", os1.tolist(), "(预期 [1, 2, 3, 3, 1, 2, 3, 1, 2])")
    print("  child2 OS =", os2.tolist(), "(预期 [2, 2, 1, 1, 3, 3, 2, 1, 3])")
    print("MS 段交叉 (J1←P2, J2←P1, J3←P2):")
    print("  child1 MS =", ms1.tolist(), "(预期 [2, 3, 1, 1, 3, 2, 1, 3, 2])")
    print("  child2 MS =", ms2.tolist(), "(预期 [3, 1, 2, 3, 2, 1, 2, 1, 3])")

    assert os1.tolist() == [1, 2, 3, 3, 1, 2, 3, 1, 2]
    assert os2.tolist() == [2, 2, 1, 1, 3, 3, 2, 1, 3]
    assert ms1.tolist() == [2, 3, 1, 1, 3, 2, 1, 3, 2]
    assert ms2.tolist() == [3, 1, 2, 3, 2, 1, 2, 1, 3]

    # 整函数 crossover()：随机多轮，子代恒合法
    rng = np.random.default_rng(0)
    n_same_os, n_same_ms = 0, 0
    n_trials = 500
    for _ in range(n_trials):
        c1, c2 = crossover(P1, P2, rng)
        assert c1.is_valid() and c2.is_valid()
        if c1.os.tolist() in (P1.os.tolist(), P2.os.tolist()):
            n_same_os += 1
        if c1.ms.tolist() in (P1.ms.tolist(), P2.ms.tolist()):
            n_same_ms += 1

    print(f"\ncrossover() 随机 {n_trials} 轮：子代全部合法 = True")
    print(f"child1 OS 恰等于某父代的比例 = {n_same_os / n_trials:.3f}")
    print(f"child1 MS 恰等于某父代的比例 = {n_same_ms / n_trials:.3f}")