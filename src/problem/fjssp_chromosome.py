"""FJSSP 解的 (OS, MS) 双向量编码数据结构（对应论文 4.3 节 Fig. 2 的表示方法）。

编码规则：
- OS（Operation Sequence）：长度为 J*M 的作业号向量，
  作业号第 k 次出现表示该作业的第 k 道操作；
- MS（Machine Sequence）：长度为 J*M 的机器号向量，
  第 j*M+1 ~ (j+1)*M 位表示作业 j+1 的第 1~M 道操作所分配的机器号。

本项目默认：每个作业的操作数相同且等于机器数 M，
并且每个作业的各道操作分配到不同的机器（即每个作业的机器号是 1..M 的一个排列）。

示例（J=3, M=3）：
    OS = (1, 2, 3, 1, 3, 2, 1, 2, 3)
    MS = (1, 2, 3, 2, 3, 1, 3, 1, 2)
    OS 对应的操作顺序为 O11-O21-O31-O12-O32-O22-O13-O23-O33，
    其中 O11 分配到机器 1、O21 分配到机器 2，以此类推。
"""

from typing import Optional, Tuple

import numpy as np


class FJSSPChromosome:
    """FJSSP 解的 (OS, MS) 编码，OS 与 MS 均为 numpy 一维整型数组。

    属性：
        os: 操作序列，存储作业号，作业号第 k 次出现代表该作业的第 k 道操作
        ms: 机器序列，按作业分组存储各操作分配到的机器号
        num_jobs: 作业数 J
        num_machines: 机器数 M（即每个作业的操作数）
        makespan: E[makespan]（评估后写入，默认 None）
        twte: E[TWTE]（评估后写入，默认 None）
        rank: 非支配前沿序号（非支配排序后写入，默认 None）
        crowding: 前沿内拥挤距离（拥挤距离分配后写入，默认 0.0）
    """

    def __init__(self, os, ms, num_machines: int):
        self.os = np.asarray(os, dtype=int)
        self.ms = np.asarray(ms, dtype=int)
        if self.os.ndim != 1 or self.ms.ndim != 1:
            raise ValueError("OS 与 MS 必须是一维向量")
        if self.os.size != self.ms.size:
            raise ValueError(
                f"OS 长度 {self.os.size} 与 MS 长度 {self.ms.size} 必须相等")
        if self.os.size % num_machines != 0:
            raise ValueError(
                f"OS 长度 {self.os.size} 必须是机器数 {num_machines} 的整数倍")
        self.num_machines = num_machines
        self.num_jobs = self.os.size // num_machines

        # 瞬时衍生状态（评估/排序阶段写入，默认表示「未评估/未排序」）
        self.makespan = None    # E[makespan]
        self.twte = None        # E[TWTE]
        self.rank = None        # 非支配前沿序号
        self.crowding = 0.0     # 前沿内拥挤距离

    # ---------- 校验 ----------

    def is_valid(self) -> bool:
        """检查解是否满足编码约束。

        约束：
        1. OS 中每个作业恰好出现 num_machines 次；
        2. MS 中每个作业的机器号是 1..num_machines 的一个排列
           （即每个作业的各操作分配到不同的机器）。
        """
        if not (1 <= self.os).all() or not (self.os <= self.num_jobs).all():
            return False
        # OS 中每个作业必须恰好出现 num_machines 次
        os_counts = np.bincount(self.os, minlength=self.num_jobs + 1)[1:]
        if not np.array_equal(os_counts, np.full(self.num_jobs, self.num_machines)):
            return False
        # MS 中每个作业的机器号必须是 1..M 的排列
        ms_blocks = self.ms.reshape(self.num_jobs, self.num_machines)
        expected = np.tile(np.arange(1, self.num_machines + 1), (self.num_jobs, 1))
        return np.array_equal(np.sort(ms_blocks, axis=1), expected)

    def validate(self) -> None:
        """校验解的合法性，不合法则抛出 ValueError。"""
        if not (1 <= self.os).all() or not (self.os <= self.num_jobs).all():
            bad = self.os[(self.os < 1) | (self.os > self.num_jobs)][0]
            raise ValueError(f"OS 中存在非法作业号 {bad}，应在 [1, {self.num_jobs}] 内")
        os_counts = np.bincount(self.os, minlength=self.num_jobs + 1)[1:]
        if not np.array_equal(os_counts, np.full(self.num_jobs, self.num_machines)):
            job = int(np.nonzero(os_counts != self.num_machines)[0][0]) + 1
            raise ValueError(
                f"OS 中作业 {job} 出现 {os_counts[job - 1]} 次，"
                f"应为 {self.num_machines} 次")
        ms_blocks = self.ms.reshape(self.num_jobs, self.num_machines)
        expected = np.tile(np.arange(1, self.num_machines + 1), (self.num_jobs, 1))
        sorted_blocks = np.sort(ms_blocks, axis=1)
        bad_rows = np.nonzero(~(sorted_blocks == expected).all(axis=1))[0]
        if bad_rows.size > 0:
            job = int(bad_rows[0]) + 1
            raise ValueError(
                f"MS 中作业 {job} 的机器分配 {ms_blocks[job - 1].tolist()} 不是 "
                f"1..{self.num_machines} 的排列")

    # ---------- 解码 ----------

    def machine_of(self, job: int, op: int) -> int:
        """返回作业 job 的第 op 道操作（均为 1 起）分配到的机器号。"""
        if not 1 <= job <= self.num_jobs:
            raise ValueError(f"作业号 {job} 超出范围 [1, {self.num_jobs}]")
        if not 1 <= op <= self.num_machines:
            raise ValueError(f"操作序号 {op} 超出范围 [1, {self.num_machines}]")
        return int(self.ms[(job - 1) * self.num_machines + (op - 1)])

    def decode(self) -> np.ndarray:
        """将解解码为 (J*M, 3) 的整型数组，列依次为 (作业号, 操作序号, 机器号)，
        行顺序即操作调度顺序。

        例如 OS=(1, 2, 3, 1, ...) 解码出的前两行为
        (1, 1, m(O11)), (2, 1, m(O21))，即先加工 O11 再加工 O21。
        """
        # occurrence就是os中每个位置上作业号是第几次出现（即该位置对应作业的第几道操作）
        # stable 排序后按作业分组、组内保持原顺序，恰好依次编号为 1..M
        occurrence = np.empty(self.os.size, dtype=int)
        occurrence[np.argsort(self.os, kind="stable")] = \
            np.tile(np.arange(1, self.num_machines + 1), self.num_jobs)
        machines = self.ms[(self.os - 1) * self.num_machines + (occurrence - 1)]
        return np.column_stack((self.os, occurrence, machines))

    # ---------- 生成 ----------

    @classmethod
    def random(cls, num_jobs: int, num_machines: int,
               rng: Optional[np.random.Generator] = None) -> "FJSSPChromosome":
        """随机生成一个合法解。

        OS 为各作业出现 M 次的随机排列，
        MS 为每个作业对机器号 1..M 的随机排列（保证各操作分配到不同机器）。
        """
        if rng is None:
            rng = np.random.default_rng()
        os_vector = rng.permutation(
            np.repeat(np.arange(1, num_jobs + 1), num_machines))
        # 逐行（逐作业）打乱机器号排列
        ms_blocks = rng.permuted(
            np.tile(np.arange(1, num_machines + 1), (num_jobs, 1)), axis=1)
        return cls(os_vector, ms_blocks.ravel(), num_machines)

    # ---------- 其他 ----------

    def copy(self) -> "FJSSPChromosome":
        clone = FJSSPChromosome(self.os.copy(), self.ms.copy(), self.num_machines)
        clone.makespan = self.makespan
        clone.twte = self.twte
        clone.rank = self.rank
        clone.crowding = self.crowding
        return clone

    def __eq__(self, other) -> bool:
        return (isinstance(other, FJSSPChromosome)
                and np.array_equal(self.os, other.os)
                and np.array_equal(self.ms, other.ms))

    def __repr__(self) -> str:
        return (f"FJSSPChromosome(num_jobs={self.num_jobs}, "
                f"num_machines={self.num_machines}, "
                f"OS={self.os.tolist()}, MS={self.ms.tolist()})")


if __name__ == "__main__":
    # 论文/需求中给出的示例：3 作业 3 机器
    example = FJSSPChromosome(
        os=[1, 2, 3, 1, 3, 2, 1, 2, 3],
        ms=[1, 2, 3, 2, 3, 1, 3, 1, 2],
        num_machines=3,
    )
    print("示例解:", example)
    print("合法解:", example.is_valid())
    print("解码结果（操作顺序 -> 机器号）:")
    for job, op, machine in example.decode():
        print(f"  O{job}{op} -> 机器 {machine}")

    # 随机生成 10-5 实例的解并校验
    sol = FJSSPChromosome.random(10, 5)
    print("\n随机生成 10 作业 5 机器的解:", sol)
    print("合法解:", sol.is_valid())

    # 非法解示例：J1 的机器分配出现重复
    bad = FJSSPChromosome(
        os=[1, 2, 2, 1], ms=[1, 1, 2, 2], num_machines=2)
    print("\n非法解 (MS=[1,1,2,2], 作业1两道操作都在机器1):", bad.is_valid())