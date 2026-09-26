"""EO 粒子数据结构，以及将 2N 维随机键向量解码为 FJSSP (OS, MS) 解的方法。

EO（Equilibrium Optimizer，Faramarzi et al.）在连续空间中演化粒子，
本模块按以下约定把连续位置映射为离散调度解：

- 粒子位置 x = [u | v]，长度 2N，N = 总工序数 = J*M（方形实例）。
  前半段 u 为排序键（决定 OS），后半段 v 为路由键（决定 MS）。
- 初始化：u、v 各维独立采样于 U(0,1)（EO 论文 Eq.6，取 C_min=0、C_max=1）。
- EO 的位置更新（Eq.16）没有任何回拉机制，迭代中键值必然出现负值和 >1 的值。
  解码只依赖键值的相对顺序、与绝对数值无关，因此不 clip、不取整——
  任何实数向量进来都解码出合法个体。

解码规则（两段式，两种不同的排序规则）：
- u → OS：全局排序（SPV，smallest position value）。
  N 个 u 键整体升序排，按排序后的位置读出各自的作业标签（键 i 属于作业 i//M+1）
  即得 OS。作业标签的出现次数 = 其工序数，多重集不变，
  因此无论键值怎么变，OS 永远是合法的作业号序列。
- v → MS：组内秩映射。
  v 按作业分组（每 M 个键一组），组内升序排，第 r 小的键 → 第 r 台机器。
  方形结构下每作业段恰好构成 {1..M} 的一个排列，MS 永远合法。

不是整个向量统一排一次序：全局排序服务全局排列约束（OS），
组内排序服务段内排列约束（MS）。

示例（J=3, M=3, N=9，粒子 18 维）：
    u = (0.30, 0.63, 0.92 | 0.18, 0.71, 0.84 | 0.05, 0.42, 0.55)
    v = (0.45, 0.20, 0.70 | 0.75, 0.10, 0.35 | 0.25, 0.50, 0.88)
    u 全局升序后 0.05（属 J3）最小 → OS = (3, 2, 1, 3, 3, 1, 2, 2, 1)
    v 组内升序排名 → MS = (2, 1, 3, 3, 1, 2, 1, 2, 3)
"""

from typing import Optional

import numpy as np

from src.data_structures.chromosome import FJSSPChromosome


def random_keys_to_chromosome(x, num_jobs: int, num_machines: int) -> FJSSPChromosome:
    """将 2N 维随机键向量解码为 FJSSP 解（N = J*M）。

    x 前半段 u 为排序键（全局 SPV → OS），后半段 v 为路由键（组内秩 → MS）。
    任何实数向量（含负值、越界值）都能解码出合法解；不做 clip。
    """
    x = np.asarray(x, dtype=float)
    num_operations = num_jobs * num_machines
    if x.ndim != 1 or x.size != 2 * num_operations:
        raise ValueError(
            f"随机键向量应为长度 {2 * num_operations} 的一维数组，实际为 shape {x.shape}")
    u, v = x[:num_operations], x[num_operations:]

    # u -> OS：全局排序（SPV）。键 i 属于作业 i//M+1，
    # 键整体升序后按顺序读出作业标签
    labels = np.repeat(np.arange(1, num_jobs + 1), num_machines)
    os_vector = labels[np.argsort(u)]

    # v -> MS：组内秩映射。第 j 组（每 M 个键）代表作业 j+1，
    # 组内第 r 小的键（即秩为 r 的键）-> 机器 r。
    # 注意 argsort 给出的是排序后各位置对应的原始下标（逆排列），
    # 而"键的秩"是它的逆映射，因此需要双重 argsort
    v_ranks = np.argsort(np.argsort(v.reshape(num_jobs, num_machines), axis=1), axis=1)
    ms_vector = (v_ranks + 1).ravel()

    return FJSSPChromosome(os_vector, ms_vector, num_machines)


class EOParticle:
    """EO 粒子：一条 2N 维连续位置向量及其评价信息。

    对应 EO 论文：粒子即解，浓度（concentration）即粒子位置（3.1 节）；
    memory saving 保存粒子历史最优坐标与适应度（3.1.5 节，类 pbest）。

    属性：
        position: 位置向量 [u | v]，前 N 维排序键、后 N 维路由键
        fitness: 当前适应度（EO 阶段为单目标 makespan），未评估时为 None
        best_position: 历史最优位置（memory saving）
        best_fitness: 历史最优适应度，未评估时为 None
        num_jobs: 作业数 J
        num_machines: 机器数 M（即每个作业的操作数）
        num_operations: 总工序数 N = J*M
    """

    def __init__(self, num_jobs: int, num_machines: int, position):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = num_jobs * num_machines
        self.position = np.asarray(position, dtype=float)
        if self.position.ndim != 1 or self.position.size != 2 * self.num_operations:
            raise ValueError(
                f"位置向量应为长度 {2 * self.num_operations} 的一维数组，"
                f"实际为 shape {self.position.shape}")
        self.fitness: Optional[float] = None
        # memory saving：初始历史最优即当前位置（论文 3.1.5 节）
        self.best_position = self.position.copy()
        self.best_fitness: Optional[float] = None

    @classmethod
    def random(cls, num_jobs: int, num_machines: int,
               rng: Optional[np.random.Generator] = None) -> "EOParticle":
        """随机生成一个粒子，位置各维独立采样于 U(0,1)（论文 Eq.6，C_min=0、C_max=1）。"""
        if rng is None:
            rng = np.random.default_rng()
        position = rng.uniform(0.0, 1.0, 2 * num_jobs * num_machines)
        return cls(num_jobs, num_machines, position)

    def to_chromosome(self) -> FJSSPChromosome:
        """解码为 FJSSP 的 (OS, MS) 染色体。"""
        return random_keys_to_chromosome(
            self.position, self.num_jobs, self.num_machines)

    def copy(self) -> "EOParticle":
        particle = EOParticle.__new__(EOParticle)
        particle.num_jobs = self.num_jobs
        particle.num_machines = self.num_machines
        particle.num_operations = self.num_operations
        particle.position = self.position.copy()
        particle.fitness = self.fitness
        particle.best_position = self.best_position.copy()
        particle.best_fitness = self.best_fitness
        return particle

    def __repr__(self) -> str:
        return (f"EOParticle(num_jobs={self.num_jobs}, "
                f"num_machines={self.num_machines}, "
                f"position={np.round(self.position, 4).tolist()}, "
                f"fitness={self.fitness})")


if __name__ == "__main__":
    # 需求中给出的示例：3 作业 3 机器，u/v 各 9 维、按作业 3 个一组
    u = [0.30, 0.63, 0.92, 0.18, 0.71, 0.84, 0.05, 0.42, 0.55]
    v = [0.45, 0.20, 0.70, 0.75, 0.10, 0.35, 0.25, 0.50, 0.88]
    particle = EOParticle(3, 3, u + v)
    chromosome = particle.to_chromosome()
    print("u =", u)
    print("v =", v)
    print("解码 OS =", chromosome.os.tolist(),
          "(预期 [3, 2, 1, 3, 3, 1, 2, 2, 1])")
    print("解码 MS =", chromosome.ms.tolist(),
          "(预期 [2, 1, 3, 3, 1, 2, 1, 2, 3])")
    print("合法解:", chromosome.is_valid())
    print("解码结果（操作顺序 -> 机器号）:")
    for job, op, machine in chromosome.decode():
        print(f"  O{job}{op} -> 机器 {machine}")

    # 随机生成 10-5 实例的粒子并校验
    sol = EOParticle.random(10, 5).to_chromosome()
    print("\n随机生成 10 作业 5 机器的粒子，解码后合法解:", sol.is_valid())

    # 含负值/越界键的向量（模拟 Eq.16 更新后的位置），解码仍应合法
    keys = np.array([1.30, -0.63, 2.92, -0.18, 1.71, 0.84, -1.05, 0.42, 1.55,
                     1.45, -0.20, 0.70, 2.75, -0.10, 0.35, -0.25, 0.50, 3.88])
    out_of_range = EOParticle(3, 3, keys).to_chromosome()
    print("\n含负值/越界键的向量解码合法（无 clip）:", out_of_range.is_valid())
    print("OS =", out_of_range.os.tolist())
    print("MS =", out_of_range.ms.tolist())
