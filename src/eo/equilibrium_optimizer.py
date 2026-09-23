"""Equilibrium Optimizer（EO）算法实现（严格按论文第 3 节及 Fig.3 伪代码）。

本实现针对 FJSSP 的 EO 阶段：粒子位置为 2N 维随机键向量（N = J*M），
解码为 (OS, MS) 染色体后以确定性 makespan 作为单目标适应度（最小化）。

对应论文公式（详见 papers/EO_pseudocode.md 与论文第 3 节）：
    f(F) = ... 适应度计算
    t     = (1 - Iter/Max_iter)^(a2 * Iter/Max_iter)                    Eq (9)
    F     = a1 * sign(r - 0.5) * [e^(-λ t) - 1]                         Eq (11)
    GCP   = 0.5*r1  (r2 >= GP)  /  0  (r2 < GP)                         Eq (15)
    G0    = GCP * (C_eq - λ * C)                                        Eq (14)
    G     = G0 * F                                                      Eq (13)
    C     = C_eq + (C - C_eq)*F + G/(λ*V) * (1 - F)                     Eq (16)

流程（Fig.3）：
    初始化种群 + 平衡候选适应度置大数；
    每代：评估全部粒子并维护四个平衡候选(C_eq1..C_eq4) → 取均值 C_ave、
    构造 equilibrium pool（5 个候选）→ memory saving（Iter>1 时）→
    计算 t → 逐粒子选候选、采样 λ/r、计算 F/GCP/G0/G、更新浓度。
"""

from typing import List, Optional

import numpy as np

from src.eo.eo_particle import EOParticle, random_keys_to_chromosome
from src.problem.fjssp_chromosome import FJSSPChromosome
from src.problem.fjssp_evaluation import compute_makespan


class EquilibriumOptimizer:
    """EO 优化器。optimize() 返回自身，之后可读取 .population / .best_fitness / .history。"""

    def __init__(self, num_jobs: int, num_machines: int, processing_times,
                 population_size: int = 100, max_iter: int = 100,
                 a1: float = 2.0, a2: float = 1.0, GP: float = 0.5,
                 rng: Optional[np.random.Generator] = None):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = num_jobs * num_machines
        self.dim = 2 * self.num_operations          # 粒子维度 2N
        self.processing_times = np.asarray(processing_times, dtype=float)
        self.population_size = population_size
        self.max_iter = max_iter
        self.a1 = a1
        self.a2 = a2
        self.GP = GP
        self.V = 1.0    # 论文 Eq.16 中 V 视为单位
        self.rng = rng if rng is not None else np.random.default_rng()

        # 运行结果
        self.population: List[EOParticle] = []
        self.best_fitness: Optional[float] = None
        self.history: List[float] = []

    # ---------- 适应度 ----------

    def _evaluate(self, particle: EOParticle) -> float:
        """确定性 makespan 作为单目标适应度（EO 阶段不开蒙特卡洛模拟）。"""
        chromosome = particle.to_chromosome()
        return compute_makespan(chromosome, self.processing_times, sim=False)

    # ---------- 主流程 ----------

    def optimize(self) -> "EquilibriumOptimizer":
        rng = self.rng
        n, dim = self.population_size, self.dim

        # 1. 初始化粒子种群（位置 ~ U(0,1)）
        population = [EOParticle.random(self.num_jobs, self.num_machines, rng)
                      for _ in range(n)]
        fitness = np.empty(n)                 # 各粒子当前适应度
        best_position = [p.best_position.copy() for p in population]   # memory saving
        best_fitness = np.full(n, np.inf)

        # 2. 平衡候选：四个 best-so-far，适应度置大数（论文 Fig.3 第 2 行）
        Ceq = np.zeros((4, dim))
        Ceq_fit = np.full(4, np.inf)

        self.history = []
        for it in range(self.max_iter):
            # ---- 评估全部粒子 + 维护四个平衡候选（if/elseif 链，Fig.3）----
            for i in range(n):
                fit_i = self._evaluate(population[i])
                fitness[i] = fit_i
                population[i].fitness = fit_i
                # 将 fit_i 按序插入四个候选（保持 Ceq_fit 升序）
                if fit_i < Ceq_fit[0]:
                    Ceq[3], Ceq_fit[3] = Ceq[2], Ceq_fit[2]
                    Ceq[2], Ceq_fit[2] = Ceq[1], Ceq_fit[1]
                    Ceq[1], Ceq_fit[1] = Ceq[0], Ceq_fit[0]
                    Ceq[0], Ceq_fit[0] = population[i].position.copy(), fit_i
                elif fit_i > Ceq_fit[0] and fit_i < Ceq_fit[1]:
                    Ceq[3], Ceq_fit[3] = Ceq[2], Ceq_fit[2]
                    Ceq[2], Ceq_fit[2] = Ceq[1], Ceq_fit[1]
                    Ceq[1], Ceq_fit[1] = population[i].position.copy(), fit_i
                elif (fit_i > Ceq_fit[0] and fit_i > Ceq_fit[1]
                      and fit_i < Ceq_fit[2]):
                    Ceq[3], Ceq_fit[3] = Ceq[2], Ceq_fit[2]
                    Ceq[2], Ceq_fit[2] = population[i].position.copy(), fit_i
                elif (fit_i > Ceq_fit[0] and fit_i > Ceq_fit[1]
                      and fit_i > Ceq_fit[2] and fit_i < Ceq_fit[3]):
                    Ceq[3], Ceq_fit[3] = population[i].position.copy(), fit_i

            # ---- C_ave 与 equilibrium pool（Eq.7）----
            Ceq_ave = Ceq.mean(axis=0)
            pool = [Ceq[0], Ceq[1], Ceq[2], Ceq[3], Ceq_ave]

            # ---- memory saving（论文 3.1.5，Iter>1 即 it>=1 时执行）----
            if it > 0:
                for i in range(n):
                    if fitness[i] < best_fitness[i]:
                        best_fitness[i] = fitness[i]
                        best_position[i] = population[i].position.copy()
                    else:
                        population[i].position = best_position[i].copy()
                        population[i].fitness = best_fitness[i]
            else:
                for i in range(n):
                    best_fitness[i] = fitness[i]
                    best_position[i] = population[i].position.copy()

            # ---- 时间因子 t（Eq.9）----
            t = (1.0 - it / self.max_iter) ** (self.a2 * it / self.max_iter)

            # ---- 逐粒子更新浓度（Eq.11/15/14/13/16）----
            for i in range(n):
                Ceq_chosen = pool[int(rng.integers(0, 5))]   # 从 5 个候选中随机选一个
                lam = rng.random(dim)        # λ ∈ [0,1)
                r = rng.random(dim)          # r ∈ [0,1)
                F = self.a1 * np.sign(r - 0.5) * (np.exp(-lam * t) - 1.0)   # Eq (11)
                r1, r2 = rng.random(), rng.random()
                GCP = 0.5 * r1 if r2 >= self.GP else 0.0                     # Eq (15)
                G0 = GCP * (Ceq_chosen - lam * population[i].position)       # Eq (14)
                G = G0 * F                                                   # Eq (13)
                C = population[i].position
                population[i].position = (Ceq_chosen + (C - Ceq_chosen) * F
                                          + G / (lam * self.V) * (1.0 - F))  # Eq (16)

            self.history.append(float(Ceq_fit[0]))

        self.population = population
        self.best_fitness = float(Ceq_fit[0])
        self.best_position = Ceq[0].copy()
        return self

    # ---------- 结果导出 ----------

    def chromosomes(self) -> List[FJSSPChromosome]:
        """将演化后的整群粒子解码为 (OS, MS) 染色体，作为 NSGA-II 初始种群。"""
        return [random_keys_to_chromosome(p.position, self.num_jobs, self.num_machines)
                for p in self.population]

    def best_chromosome(self) -> FJSSPChromosome:
        """最优粒子（平衡候选 Ceq1）解码出的染色体。"""
        return random_keys_to_chromosome(self.best_position, self.num_jobs, self.num_machines)


if __name__ == "__main__":
    # 小型验证：10 作业 5 机器，随机加工时间矩阵，跑 EO 看适应度下降
    rng = np.random.default_rng(0)
    pt = rng.integers(1, 10, size=(10, 5)).astype(float)

    eo = EquilibriumOptimizer(
        num_jobs=10, num_machines=5, processing_times=pt,
        population_size=40, max_iter=60, rng=np.random.default_rng(1))
    eo.optimize()

    print("EO 最优 makespan =", round(eo.best_fitness, 4))
    print("收敛过程（每代最优）:",
          [round(v, 1) for v in eo.history[::10]])

    chromosomes = eo.chromosomes()
    valid = all(c.is_valid() for c in chromosomes)
    print("整群解码染色体数 =", len(chromosomes), "，全部合法 =", valid)
    best = eo.best_chromosome()
    print("最优染色体 =", best)
    print("最优染色体的确定性 makespan =",
          compute_makespan(best, pt, sim=False))