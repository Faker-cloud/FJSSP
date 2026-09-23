"""NSGA-II 二元锦标赛选择。

规则（标准 NSGA-II）：随机抽两个不同个体，按字典序 (rank 升序, crowding 降序) 取胜者——
    1. 非支配层级 rank 小者胜（前沿靠前更优）；
    2. rank 相同时拥挤距离 crowding 大者胜（保持多样性）。

读取种群个体的 `.rank` / `.crowding` 属性，返回选中的个体对象（供交叉使用）。

时序约定：进入本模块前，种群已完成非支配排序 + 拥挤距离分配，
即每个 FJSSPChromosome 的 `.rank` / `.crowding` 已赋真值
（初始种群在首代选择前也需先排序 + 算拥挤一次）。
"""

from typing import List, Optional, Tuple

import numpy as np

from src.problem.fjssp_chromosome import FJSSPChromosome


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
    print("全部断言通过")