"""NSGA-II 非支配排序 + 拥挤距离分配（Deb et al. 2002）。

两个确定性原语（无随机数），均从个体 `.makespan` / `.twte` 属性读双目标
（两目标均最小化）、就地写 `.rank` / `.crowding`：

- `fast_non_dominated_sort(population)`：快排非支配排序，逐个前沿赋 `.rank`
  （0 起，rank=0 为第一前沿 F1），返回按前沿升序排列的前沿列表。
- `crowding_distance_assignment(front)`：对单个前沿赋 `.crowding`
  （归一化版本，边界点 = +inf）。

主循环用法：
    fronts = fast_non_dominated_sort(R)       # R = 父代 ∪ 子代（已短模拟评估）
    for fr in fronts:
        crowding_distance_assignment(fr)
    P = select_best(fronts, N)                # from src.nsga2.selection
"""

from typing import List

import numpy as np

from src.problem.fjssp_chromosome import FJSSPChromosome


def _objective_matrix(population: List[FJSSPChromosome]) -> np.ndarray:
    """抽出 (N, 2) 目标矩阵，列 = (makespan, twte)；未评估（None）则报错。"""
    makespans = []
    twtes = []
    for idx, ind in enumerate(population):
        if ind.makespan is None or ind.twte is None:
            raise ValueError(
                f"个体 {idx} 的目标值未评估（makespan/twte 为 None），"
                "请先对种群做短模拟评估")
        makespans.append(ind.makespan)
        twtes.append(ind.twte)
    return np.column_stack((np.asarray(makespans, dtype=float),
                            np.asarray(twtes, dtype=float)))


def _dominance_matrix(F: np.ndarray) -> np.ndarray:
    """返回布尔矩阵 dominates[i, j]=真 表示解 i 支配解 j（两目标均最小化）。"""
    le = F[:, None, :] <= F[None, :, :]   # i 各目标不劣于 j
    lt = F[:, None, :] < F[None, :, :]    # i 至少一个目标严格优于 j
    return np.all(le, axis=2) & np.any(lt, axis=2)


def fast_non_dominated_sort(population: List[FJSSPChromosome]
                            ) -> List[List[FJSSPChromosome]]:
    """快排非支配排序：给每个个体就地赋 `.rank`（0 起），返回前沿列表（按 rank 升序）。"""
    n = len(population)
    if n == 0:
        return []
    F = _objective_matrix(population)
    dominates = _dominance_matrix(F)
    dominated_count = dominates.sum(axis=0)   # 支配每个解的个体数

    fronts: List[List[FJSSPChromosome]] = []
    assigned = np.zeros(n, dtype=bool)
    rank = 0
    while not assigned.all():
        current = np.nonzero((dominated_count == 0) & ~assigned)[0]
        for p in current:
            population[int(p)].rank = rank
        fronts.append([population[int(p)] for p in current])
        assigned[current] = True
        for p in current:
            dominated_count[dominates[p]] -= 1
        rank += 1
    return fronts


def crowding_distance_assignment(front: List[FJSSPChromosome]) -> None:
    """对单个前沿就地赋 `.crowding`（Deb 2002 归一化，边界点 = +inf）。"""
    m = len(front)
    if m == 0:
        return
    F = _objective_matrix(front)
    crowd = np.zeros(m)
    for k in range(F.shape[1]):
        f = F[:, k]
        order = np.argsort(f)
        crowd[order[0]] = np.inf
        crowd[order[-1]] = np.inf
        span = f[order[-1]] - f[order[0]]
        if span > 0:
            crowd[order[1:-1]] += (f[order[2:]] - f[order[:-2]]) / span
    for i in range(m):
        front[i].crowding = float(crowd[i])


if __name__ == "__main__":
    def _make(makespan, twte) -> FJSSPChromosome:
        c = FJSSPChromosome(
            os=[1, 2, 2, 3, 1, 3, 2, 1, 3],
            ms=[3, 1, 2, 1, 3, 2, 2, 1, 3], num_machines=3)
        c.makespan = makespan
        c.twte = twte
        return c

    # 1. 非支配排序手工用例：A(1,1) B(2,2) C(3,1) D(2,3) E(4,4)
    pts = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.0), (2.0, 3.0), (4.0, 4.0)]
    pop = [_make(m, t) for m, t in pts]
    fronts = fast_non_dominated_sort(pop)
    got = [[(ind.makespan, ind.twte) for ind in fr] for fr in fronts]
    print("前沿划分:", got)
    assert [ind.rank for ind in pop] == [0, 1, 1, 2, 3]
    assert got == [[(1.0, 1.0)], [(2.0, 2.0), (3.0, 1.0)],
                   [(2.0, 3.0)], [(4.0, 4.0)]]

    # 2. 支配矩阵与朴素 O(N^2) 交叉验证
    F = _objective_matrix(pop)
    D = _dominance_matrix(F)
    n = len(pop)
    D_naive = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            D_naive[i, j] = (F[i] <= F[j]).all() and (F[i] < F[j]).any()
    assert np.array_equal(D, D_naive)
    print("支配矩阵交叉验证一致")

    # 3. 拥挤距离手工用例：单前沿 [(1,3),(2,2),(3,1)]，中间点 = 2.0
    front3 = [_make(m, t) for m, t in [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]]
    crowding_distance_assignment(front3)
    cw = [ind.crowding for ind in front3]
    print("拥挤距离:", cw, "(预期 [inf, 2.0, inf])")
    assert cw[0] == float("inf") and cw[2] == float("inf")
    assert cw[1] == 2.0

    # 4. 退化：单点/两点全 inf；span==0 不除零
    s1 = [_make(5.0, 5.0)]
    crowding_distance_assignment(s1)
    assert s1[0].crowding == float("inf")
    s2 = [_make(5.0, 5.0), _make(5.0, 5.0)]
    crowding_distance_assignment(s2)
    assert all(x.crowding == float("inf") for x in s2)
    s0 = [_make(7.0, 7.0), _make(7.0, 7.0), _make(7.0, 7.0)]
    crowding_distance_assignment(s0)    # 不触发除零
    print("退化用例通过")

    # 5. 与 select_best 衔接：排序 + 拥挤 → 前沿直传截断
    from src.nsga2.selection import select_best
    pop2 = [_make(m, t) for m, t in [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0),   # F0
                                     (2.0, 5.0), (3.0, 4.0), (4.0, 3.0)]]  # F1
    fronts2 = fast_non_dominated_sort(pop2)
    for fr in fronts2:
        crowding_distance_assignment(fr)
    sel = select_best(fronts2, 5)      # F0 全 3 个 + F1 按拥挤取前 2 个
    assert len(sel) == 5
    assert [x.rank for x in sel] == [0, 0, 0, 1, 1]
    print("sort+crowd+select_best 衔接通过，选中 5 个：",
          [(x.makespan, x.twte, x.rank) for x in sel])

    # 6. None 守卫：未评估染色体不可排序
    fresh = FJSSPChromosome(
        os=[1, 2, 2, 3, 1, 3, 2, 1, 3],
        ms=[3, 1, 2, 1, 3, 2, 2, 1, 3], num_machines=3)
    try:
        fast_non_dominated_sort([fresh])
        raise SystemExit("应已抛出 ValueError")
    except ValueError:
        pass
    print("\n全部断言通过")