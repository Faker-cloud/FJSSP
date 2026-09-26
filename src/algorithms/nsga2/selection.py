"""NSGA-II 选择：父代二元锦标赛 + 环境选择。

- 二元锦标赛（供交叉选父代）：随机抽两个不同个体，按字典序
  (rank 升序, crowding 降序) 取胜者——
    1. 非支配层级 rank 小者胜（前沿靠前更优）；
    2. rank 相同时拥挤距离 crowding 大者胜（保持多样性）。
- 环境选择（select_best，供下一代精英保留）：直接消费 fast_non_dominated_sort
  返回的前沿列表，整前沿保留 + 对装不下的末前沿按拥挤距离降序截断，选出规模 n 的个体。

两个入口都读取种群个体的 `.rank` / `.crowding` 属性。

时序约定：进入本模块前，种群已完成非支配排序 + 拥挤距离分配，
即每个 FJSSPChromosome 的 `.rank` / `.crowding` 已赋真值
（初始种群在首代选择前也需先排序 + 算拥挤一次）。
"""

from typing import List, Optional, Tuple

import numpy as np

from src.data_structures.chromosome import FJSSPChromosome


def _is_better(rank_a, rank_b, crowd_a, crowd_b) -> bool:
    """候选 a 是否优于候选 b：rank 小优先；rank 平则 crowding 大优先。"""
    return (rank_a < rank_b) or (rank_a == rank_b and crowd_a > crowd_b)


def _ranks_crowding(population):
    """把种群个体的 rank / crowding 摊平成两个 numpy 数组（与种群下标一一对应）。"""
    ranks = np.array([ind.rank for ind in population], dtype=float)
    crowding = np.array([ind.crowding for ind in population], dtype=float)
    return ranks, crowding


def tournament_select(population: List[FJSSPChromosome], num_matches: int,
                      rng: Optional[np.random.Generator] = None) -> List[FJSSPChromosome]:
    """批量二元锦标赛：返回 num_matches 个胜者对象，每场独立抽两个不同个体。"""
    n = len(population)
    if n < 2:
        raise ValueError("二元锦标赛需要至少 2 个个体")
    if rng is None:
        rng = np.random.default_rng()

    ranks, crowding = _ranks_crowding(population)
    a = rng.integers(0, n, size=num_matches)
    b = rng.integers(0, n - 1, size=num_matches)
    b = np.where(b >= a, b + 1, b)          # 保证每场内 b != a
    better_is_a = (ranks[a] < ranks[b]) | \
                  ((ranks[a] == ranks[b]) & (crowding[a] > crowding[b]))
    winners = np.where(better_is_a, a, b)
    return [population[int(i)] for i in winners]


def select_parents(population: List[FJSSPChromosome], num_pairs: int,
                   rng: Optional[np.random.Generator] = None
                   ) -> List[Tuple[FJSSPChromosome, FJSSPChromosome]]:
    """为交叉选父代：先随机打乱种群顺序，再做 2*num_pairs 场独立锦标赛、按行配对。

    打乱用于消除「非支配排序 → 环境选择」产生的结构化顺序对平局打破/顺序配对的潜在影响。
    返回 num_pairs 个 (父代1, 父代2) 元组；同一对两个父代可能相同（自交），交叉仍合法。
    """
    if rng is None:
        rng = np.random.default_rng()
    shuffled = [population[i] for i in rng.permutation(len(population))]
    winners = tournament_select(shuffled, 2 * num_pairs, rng)
    return list(zip(winners[0::2], winners[1::2]))


def select_best(fronts: List[List[FJSSPChromosome]], n: int) -> List[FJSSPChromosome]:
    """NSGA-II 环境选择：从前沿列表（fast_non_dominated_sort 的返回值）中
    选出规模 n 的下一代（精英保留）。

    规则（Algorithm 1「按 (前沿等级, 拥挤距离) 从 R 选最优 N 个」）：
        1. 从最前的前沿起，逐个整前沿保留；
        2. 当整前沿装不下（再加该前沿会超过 n）时，对该前沿按拥挤距离降序
           截断，取恰好凑满 n 所需的个数（截断前沿内拥挤大者胜出）。

    预条件：fronts 为按前沿升序排列的列表（同一前沿个体相邻、列表结构即显式
    分组，无需 rank 值）；各前沿须已完成 crowding_distance_assignment，
    否则末前沿截断退化为按输入顺序。
    返回长度 n 的新列表（元素为原个体对象引用），不改动输入。
    """
    n_total = sum(len(fr) for fr in fronts)
    if not 0 <= n <= n_total:
        raise ValueError(f"n={n} 应满足 0 <= n <= 种群规模 {n_total}")

    selected: List[FJSSPChromosome] = []
    for fr in fronts:
        if len(selected) + len(fr) <= n:
            selected.extend(fr)
        else:
            need = n - len(selected)
            fr_sorted = sorted(fr, key=lambda c: c.crowding, reverse=True)
            selected.extend(fr_sorted[:need])
            break
    return selected


if __name__ == "__main__":
    def make(rank, crowding) -> FJSSPChromosome:
        c = FJSSPChromosome(
            os=[1, 2, 2, 3, 1, 3, 2, 1, 3],
            ms=[3, 1, 2, 1, 3, 2, 2, 1, 3], num_machines=3)
        c.rank = rank
        c.crowding = crowding
        return c

    # 1. 新构造染色体的 4 个衍生属性默认值（未评估/未排序状态）
    fresh = FJSSPChromosome(
        os=[1, 2, 2, 3, 1, 3, 2, 1, 3],
        ms=[3, 1, 2, 1, 3, 2, 2, 1, 3], num_machines=3)
    assert fresh.makespan is None and fresh.twte is None
    assert fresh.rank is None and fresh.crowding == 0.0
    print("新染色体衍生属性 (makespan, twte, rank, crowding) =",
          (fresh.makespan, fresh.twte, fresh.rank, fresh.crowding))

    # 2. _is_better 真值表
    assert _is_better(0, 1, 0.0, 0.0) is True    # rank 小者胜
    assert _is_better(1, 0, 0.0, 0.0) is False
    assert _is_better(1, 1, 5.0, 1.0) is True    # rank 平、crowding 大者胜
    assert _is_better(1, 1, 5.0, 5.0) is False   # 完全平判负
    print("_is_better 真值表通过")

    # 3. n=2 确定性用例（候选必为 {0,1}，恒选唯一较优者）
    rng = np.random.default_rng(0)
    pop2 = [make(0, 0.0), make(1, 0.0)]
    assert tournament_select(pop2, 1, rng)[0] is pop2[0]     # rank 优
    pop2b = [make(1, 5.0), make(1, 1.0)]
    assert tournament_select(pop2b, 1, rng)[0] is pop2b[0]   # rank 平、crowding 优
    print("n=2 确定性用例通过")

    # 4. 选择压力统计（按对象身份计数，避免 __eq__ 只比 os/ms）
    ranks = [0, 1, 1, 1, 1, 1]
    crowds = [0.0, 9.0, 7.0, 5.0, 3.0, 1.0]
    pop = [make(r, c) for r, c in zip(ranks, crowds)]
    rng2 = np.random.default_rng(1)
    winners = tournament_select(pop, 100000, rng2)
    id_to_idx = {id(p): i for i, p in enumerate(pop)}
    counts = np.zeros(len(pop), dtype=int)
    for w in winners:
        counts[id_to_idx[id(w)]] += 1
    total = len(winners)
    print("\n选择压力（rank=[0,1,1,1,1,1]，crowding 递减）胜出占比:")
    for i, p in enumerate(pop):
        print(f"  个体{i} (rank={p.rank}, crowding={p.crowding:.1f}): "
              f"{counts[i] / total:.3f}")
    assert counts[0] > counts[1]
    assert counts[1] >= counts[2] >= counts[3] >= counts[4] >= counts[5]

    # 5. select_parents 形状与合法性
    rng3 = np.random.default_rng(2)
    pairs = select_parents(pop, 50, rng3)
    assert len(pairs) == 50
    assert all(any(p is q for q in pop) and any(p2 is q for q in pop)
               for p, p2 in pairs)
    print(f"\nselect_parents 返回 {len(pairs)} 对父代，元素均为种群同引用对象")

    # 6. select_best 环境选择：整前沿保留 + 末前沿按拥挤距离降序截断
    fronts = [
        [make(0, c) for c in [0.0, 0.0]],
        [make(1, c) for c in [9.0, 7.0, 5.0]],
        [make(2, c) for c in [8.0, 6.0, 4.0]],
    ]
    sel5 = select_best(fronts, 5)                   # 前沿0(2) + 前沿1(3) 整前沿
    assert [ind.rank for ind in sel5] == [0, 0, 1, 1, 1]
    sel4 = select_best(fronts, 4)                   # 前沿1 截断：取 crowding 9,7
    assert [ind.crowding for ind in sel4] == [0.0, 0.0, 9.0, 7.0]
    sel7 = select_best(fronts, 7)                   # 前沿2 截断：取 crowding 8,6
    assert [ind.crowding for ind in sel7] == [0.0, 0.0, 9.0, 7.0, 5.0, 8.0, 6.0]
    assert select_best(fronts, 0) == []
    total = sum(len(fr) for fr in fronts)
    assert len(select_best(fronts, total)) == total
    print("\nselect_best 环境选择通过（fronts 直传，整前沿保留 + 末前沿按拥挤截断）")
    print("全部断言通过")