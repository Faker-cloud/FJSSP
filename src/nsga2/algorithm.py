"""SIM-NSGA-II 主循环（论文 Algorithm 1）。

把 NSGA-II 各算子串成完整演化流程：
    初始化（随机 / EO 整群解码，可切换）→ 短模拟评估 →
    每代：选父代 → 交叉(cr)/复制 → 变异(mr) → 子代短模拟评估 →
          合并 2N → 非支配排序 + 拥挤距离 → 环境选择 N →
    末代：长模拟重评价 → 重排序取 Pareto 前沿。

三个实现约定（用户要求）：
1. 初始化预留接口：构造参数 initial_population 填 None 即随机初始化（基线 Sim-NSGA-II），
   填 EO 整群解码的染色体即 EO-Sim-NSGA-II；由调用方把两者接起来，主模块不与 EO 硬耦合。
2. 评估跟随 sim 开关：sim=True 走蒙特卡洛（短模拟 s_short、末代长模拟 s_long），
   sim=False 走确定性单次评估。
3. 不做种群修复：交叉/变异/随机/EO 解码都保证染色体合法，主循环不调用任何修复逻辑。
"""

from typing import List, Optional

import numpy as np

from src.problem.fjssp_chromosome import FJSSPChromosome
from src.problem.fjssp_evaluation import compute_objectives
from src.nsga2.crossover import crossover
from src.nsga2.mutation import mutate
from src.nsga2.selection import select_parents, select_best
from src.nsga2.sorting import fast_non_dominated_sort, crowding_distance_assignment


class SimNSGAII:
    """SIM-NSGA-II 优化器。optimize() 返回自身，结果读 population / pareto_front / history。"""

    def __init__(self, num_jobs: int, num_machines: int, processing_times,
                 priorities, due_dates, population_size: int, num_generations: int,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1,
                 uncertain_level: float = 0.1, sim: bool = True,
                 s_short: int = 20, s_long: int = 1000,
                 initial_population: Optional[List[FJSSPChromosome]] = None,
                 rng: Optional[np.random.Generator] = None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = np.asarray(processing_times, dtype=float)
        self.priorities = np.asarray(priorities, dtype=float)
        self.due_dates = np.asarray(due_dates, dtype=float)
        if self.processing_times.shape != (num_jobs, num_machines):
            raise ValueError(
                f"processing_times 形状应为 {(num_jobs, num_machines)}，"
                f"实际为 {self.processing_times.shape}")
        if self.priorities.shape != (num_jobs,) or self.due_dates.shape != (num_jobs,):
            raise ValueError(
                f"priorities 与 due_dates 应为长度 {num_jobs} 的一维数组，"
                f"实际为 {self.priorities.shape} 与 {self.due_dates.shape}")
        if np.any(self.priorities <= 0):
            raise ValueError("优先级必须为正数（提前权重为其倒数）")

        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.uncertain_level = uncertain_level
        self.sim = sim
        self.s_short = s_short
        self.s_long = s_long
        self.initial_population = initial_population
        self.rng = rng if rng is not None else np.random.default_rng()

        # 运行结果（optimize() 后填充）
        self.population: List[FJSSPChromosome] = []
        self.pareto_front: List[FJSSPChromosome] = []
        self.history: List[int] = []               # 每代 F0 规模（合并 2N 口径）
        self.history_makespan: List[float] = []    # 每代合并种群的最优 makespan（收敛图）
        self.history_twte: List[float] = []        # 每代合并种群的最优 twte（收敛图）

    # ---------- 初始化（预留接口） ----------

    def _initialize(self) -> List[FJSSPChromosome]:
        n = self.population_size
        if self.initial_population is None:
            return [FJSSPChromosome.random(self.num_jobs, self.num_machines, self.rng)
                    for _ in range(n)]
        if len(self.initial_population) != n:
            raise ValueError(
                f"初始种群规模应为 {n}，实际为 {len(self.initial_population)}")
        return list(self.initial_population)

    # ---------- 评估（跟随 sim 开关） ----------

    def _evaluate(self, chromosome: FJSSPChromosome, num_samples: int) -> None:
        makespan, twte = compute_objectives(
            chromosome, self.processing_times, self.priorities, self.due_dates,
            sim=self.sim, uncertain_level=self.uncertain_level,
            num_samples=num_samples, rng=self.rng)
        chromosome.makespan = makespan
        chromosome.twte = twte

    def _evaluate_population(self, population: List[FJSSPChromosome],
                             num_samples: int) -> None:
        for chromosome in population:
            self._evaluate(chromosome, num_samples)

    # ---------- 生成子代（cr / mr，无修复） ----------

    def _generate_offspring(self, parents: List[FJSSPChromosome]
                            ) -> List[FJSSPChromosome]:
        n = self.population_size
        num_pairs = (n + 1) // 2
        pairs = select_parents(parents, num_pairs, self.rng)
        offspring: List[FJSSPChromosome] = []
        for p1, p2 in pairs:
            if self.rng.random() < self.crossover_rate:
                c1, c2 = crossover(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = mutate(c1, self.mutation_rate, self.rng)
            c2 = mutate(c2, self.mutation_rate, self.rng)
            offspring.extend([c1, c2])
        return offspring[:n]     # N 为奇数时截断到恰好 N

    # ---------- 主循环 ----------

    def optimize(self) -> "SimNSGAII":
        n, g = self.population_size, self.num_generations

        population = self._initialize()
        self._evaluate_population(population, self.s_short)
        # 首代锦标赛选择的前置条件：初始种群先排序 + 算拥挤（时序约定见 selection.py）
        fronts = fast_non_dominated_sort(population)
        for fr in fronts:
            crowding_distance_assignment(fr)

        self.history = []
        self.history_makespan = []
        self.history_twte = []
        for _ in range(g):
            offspring = self._generate_offspring(population)
            self._evaluate_population(offspring, self.s_short)

            merged = population + offspring                     # 规模 2N
            fronts = fast_non_dominated_sort(merged)
            for fr in fronts:
                crowding_distance_assignment(fr)
            population = select_best(fronts, n)
            self.history.append(len(fronts[0]))
            # 收敛序列：环境选择后存活种群(N)的最优——只有被选中的个体才是该代得到的解
            self.history_makespan.append(min(c.makespan for c in population))
            self.history_twte.append(min(c.twte for c in population))

        # 末代长模拟重评价，并重排序取最终 Pareto 前沿
        self._evaluate_population(population, self.s_long)
        fronts = fast_non_dominated_sort(population)
        for fr in fronts:
            crowding_distance_assignment(fr)

        self.population = population
        self.pareto_front = fronts[0] if fronts else []
        return self


if __name__ == "__main__":
    def make_instance(j, m, rng):
        pt = rng.integers(1, 10, size=(j, m)).astype(float)
        priorities = rng.integers(1, 5, size=j).astype(float)
        due_dates = rng.integers(5, 40, size=j).astype(float)
        return pt, priorities, due_dates

    def all_valid(pop):
        return all(c.is_valid() for c in pop)

    rng0 = np.random.default_rng(0)
    pt, pr, dd = make_instance(3, 3, rng0)
    J = M = 3

    # 1. 确定性跑通（sim=False）
    alg = SimNSGAII(J, M, pt, pr, dd, population_size=20, num_generations=5,
                    sim=False, rng=np.random.default_rng(1)).optimize()
    assert len(alg.population) == 20 and all_valid(alg.population)
    assert len(alg.pareto_front) >= 1
    assert all(np.isfinite(c.makespan) and np.isfinite(c.twte) for c in alg.population)
    assert len(alg.history) == 5
    assert len(alg.history_makespan) == 5 and len(alg.history_twte) == 5
    assert all(np.isfinite(alg.history_makespan))
    assert all(np.isfinite(alg.history_twte))
    print("确定性跑通：population=20，pareto_front=", len(alg.pareto_front),
          "，history=", alg.history)

    # 2. MC 跑通（sim=True，短/长模拟分别用 s_short/s_long）
    alg2 = SimNSGAII(J, M, pt, pr, dd, population_size=12, num_generations=3,
                     sim=True, s_short=10, s_long=50,
                     rng=np.random.default_rng(2)).optimize()
    assert all(np.isfinite(c.makespan) and np.isfinite(c.twte) for c in alg2.population)
    print("MC 跑通：pareto_front[0] makespan =",
          round(alg2.pareto_front[0].makespan, 3),
          "twte =", round(alg2.pareto_front[0].twte, 3))

    # 3. 初始化接口（要求 1）
    # 3b. 传入预构建的合法种群
    init = [FJSSPChromosome.random(J, M, np.random.default_rng(i)) for i in range(20)]
    alg3 = SimNSGAII(J, M, pt, pr, dd, population_size=20, num_generations=2,
                     sim=False, initial_population=init,
                     rng=np.random.default_rng(3)).optimize()
    assert len(alg3.population) == 20 and all_valid(alg3.population)
    # 3c. EO 整群解码 → initial_population 交接
    from src.eo.equilibrium_optimizer import EquilibriumOptimizer
    eo = EquilibriumOptimizer(J, M, pt, population_size=20, max_iter=30,
                              rng=np.random.default_rng(4)).optimize()
    eo_init = eo.chromosomes()
    assert all_valid(eo_init)
    alg4 = SimNSGAII(J, M, pt, pr, dd, population_size=20, num_generations=2,
                     sim=False, initial_population=eo_init,
                     rng=np.random.default_rng(5)).optimize()
    assert all_valid(alg4.population)
    # 3d. 规模 ≠ N 报错
    try:
        SimNSGAII(J, M, pt, pr, dd, population_size=20, num_generations=1,
                  sim=False, initial_population=init[:5],
                  rng=np.random.default_rng(6)).optimize()
        raise SystemExit("应已抛 ValueError")
    except ValueError:
        pass
    print("初始化接口通过（random / 预构建 / EO 交接 / 规模校验）")

    # 4. 子代合法（要求 3 佐证）：无任何修复，仍全合法
    alg5 = SimNSGAII(J, M, pt, pr, dd, population_size=10, num_generations=1,
                     sim=False, rng=np.random.default_rng(7))
    parents = [FJSSPChromosome.random(J, M, alg5.rng) for _ in range(10)]
    off = alg5._generate_offspring(parents)
    assert len(off) == 10 and all_valid(off)
    print("子代合法：_generate_offspring 返回 10 个、全部合法（无修复）")

    # 5. Pareto 前沿两两互不支配
    def is_dominated_by(a, b):
        return (b.makespan <= a.makespan and b.twte <= a.twte) and \
               (b.makespan < a.makespan or b.twte < a.twte)
    pf = alg.pareto_front
    for i in range(len(pf)):
        for j in range(len(pf)):
            assert not is_dominated_by(pf[i], pf[j]), "Pareto 前沿成员应互不支配"
    print("Pareto 前沿互不支配：", len(pf), "个成员通过")

    # 6. cr=0 退化：cr=0、mr=0 时子代即父代副本
    alg6 = SimNSGAII(J, M, pt, pr, dd, population_size=10, num_generations=1,
                     sim=False, crossover_rate=0.0, mutation_rate=0.0,
                     rng=np.random.default_rng(8))
    par = [FJSSPChromosome.random(J, M, alg6.rng) for _ in range(10)]
    off6 = alg6._generate_offspring(par)
    assert all(any(c == p for p in par) for c in off6)
    print("cr=0 退化：子代均为父代副本（未交叉、未变异）")

    # 7. 首代选择前初始种群已完成排序 + 算拥挤（回归测试：修复前 rank=None 会失败）
    class _SpyNSGAII(SimNSGAII):
        def _generate_offspring(self, parents):
            if not hasattr(self, "_first_parents"):
                self._first_parents = list(parents)
            return super()._generate_offspring(parents)

    spy = _SpyNSGAII(J, M, pt, pr, dd, population_size=10, num_generations=1,
                     sim=False, rng=np.random.default_rng(9)).optimize()
    assert all(c.rank is not None for c in spy._first_parents)
    assert all(np.isfinite(c.crowding) or c.crowding == float("inf")
               for c in spy._first_parents)
    print("首代选择前初始种群已排序 + 算拥挤（rank/crowding 全部就绪）")

    print("\n全部断言通过")