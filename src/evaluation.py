"""根据 FJSSP 解（FJSSPChromosome）计算 makespan 与 TWTE，支持蒙特卡洛模拟开关。

调度构造规则（semi-active 贪心插入）：
按 OS 顺序逐操作调度，每个操作的开工时间为
    start = max(作业上一道工序的完工时间, 所分配机器的就绪时间)，
finish = start + pt[job, op]，并同时更新作业就绪与机器就绪时间；
makespan = 所有作业完工时间的最大值。

随机加工时间（论文 3.2 节，Eq.11-14）：
每个操作的加工时间 SPt_ij 服从对数正态分布，其均值等于数据集给定的平均加工时间
    E[SPt_ij] = pt_ij                        (Eq.11)
    Var[SPt_ij] = UL * E[SPt_ij]             (Eq.12，UL 为不确定性水平)
    mu_ij = ln E - 1/2 * ln(1 + Var/E^2)     (Eq.13)
    sigma_ij = |sqrt(ln(1 + Var/E^2))|       (Eq.14)

TWTE（论文 3.1 节，Eq.1/3/4）：
    TWTE = sum_i ( WT_i * T_i + WE_i * E_i )          (Eq.1)
    T_i = max(0, 作业完工时间 - 交期 d_i)              (Eq.3)
    E_i = max(0, 交期 d_i - 作业完工时间)              (Eq.4)
权重沿用旧 sim_mo_jsp.py 的约定：拖延权重 WT_i = 优先级 priority_i，
提前权重 WE_i = 1/priority_i（priority 与交期来自数据集「Priority and Due date」表）。

蒙特卡洛模拟（论文 Algorithm 2）：
每轮对所有操作各采一个加工时间构成一个完整场景，调度一次得到该场景的目标值；
共 n 轮，最终目标值取 n 轮结果的平均（先算 n 次目标值再平均）。
"""

from typing import Optional, Tuple

import numpy as np

from src.data_structures.chromosome import FJSSPChromosome


def lognormal_params(mean_pt, uncertain_level: float) -> Tuple[np.ndarray, np.ndarray]:
    """由平均加工时间矩阵与不确定性水平 UL 计算对数正态分布参数（论文 Eq.12-14）。

    mean_pt: (J, M) 各操作的平均加工时间（即分布均值，Eq.11）
    返回 (mu, sigma)：ln(SPt) 的均值矩阵与标准差矩阵
    """
    mean_pt = np.asarray(mean_pt, dtype=float)
    var = uncertain_level * mean_pt    # Eq.12: Var = UL * E[SPt]
    sigma_sq = np.log(1 + var / mean_pt ** 2)   # ln(SPt) 的方差
    mu = np.log(mean_pt) - sigma_sq / 2         # ln(SPt) 的均值
    return mu, np.sqrt(sigma_sq)


def sample_scenario(mean_pt, uncertain_level: float,
                    rng: np.random.Generator) -> np.ndarray:
    """采样一个完整场景：对每个操作从其对数正态分布各采一个加工时间。"""
    mu, sigma = lognormal_params(mean_pt, uncertain_level)
    return rng.lognormal(mean=mu, sigma=sigma)


def _schedule_job_completions(ops: np.ndarray, processing_times: np.ndarray) -> np.ndarray:
    """按 semi-active 规则调度，返回各作业的完工时间数组 (J,)。

    ops: chromosome.decode() 得到的 (N, 3) 数组，列依次为 (作业号, 操作序号, 机器号)，均 1 起
    processing_times: (J, M) 加工时间矩阵，pt[j-1, op-1] 为该操作的加工时间
    """
    num_jobs = processing_times.shape[0]
    num_machines = processing_times.shape[1]
    job_ready = np.zeros(num_jobs)       # 各作业上一道工序的完工时间
    machine_ready = np.zeros(num_machines)   # 各机器的就绪时间
    for job, op, machine in ops:
        j, o, m = job - 1, op - 1, machine - 1
        start = max(job_ready[j], machine_ready[m])
        finish = start + processing_times[j, o]
        job_ready[j] = finish
        machine_ready[m] = finish
    return job_ready


def schedule_trace(chromosome: FJSSPChromosome, processing_times) -> np.ndarray:
    """按 semi-active 贪心规则排程，返回完整调度轨迹。

    返回 (N, 5) 数组，列依次为 (作业号, 操作序号, 机器号, 开工时间, 完工时间)，均 1 起，
    行顺序即调度顺序（同 chromosome.decode()）。仅甘特图等展示用途，MC 评估热路径不用。
    """
    processing_times = np.asarray(processing_times, dtype=float)
    ops = chromosome.decode()
    num_jobs = chromosome.num_jobs
    num_machines = chromosome.num_machines
    job_ready = np.zeros(num_jobs)
    machine_ready = np.zeros(num_machines)
    starts = np.zeros(ops.shape[0])
    finishes = np.zeros(ops.shape[0])
    for idx, (job, op, machine) in enumerate(ops):
        j, o, m = job - 1, op - 1, machine - 1
        start = max(job_ready[j], machine_ready[m])
        finish = start + processing_times[j, o]
        starts[idx] = start
        finishes[idx] = finish
        job_ready[j] = finish
        machine_ready[m] = finish
    return np.column_stack((ops, starts, finishes))


def _schedule_makespan(ops: np.ndarray, processing_times: np.ndarray) -> float:
    """按 semi-active 规则调度，返回 makespan（= max 各作业完工时间）。"""
    return float(_schedule_job_completions(ops, processing_times).max())


def _twte_from_completions(job_completions: np.ndarray, priorities: np.ndarray,
                           due_dates: np.ndarray) -> float:
    """由各作业完工时间计算 TWTE（论文 Eq.1/3/4）。

    job_completions: (J,) 各作业完工时间
    priorities: (J,) 各作业优先级，拖延权重 WT_i = priority_i，提前权重 WE_i = 1/priority_i
    due_dates: (J,) 各作业交期
    """
    tardiness = np.maximum(0.0, job_completions - due_dates)
    earliness = np.maximum(0.0, due_dates - job_completions)
    return float(np.sum(priorities * tardiness + (1.0 / priorities) * earliness))


def compute_makespan(chromosome: FJSSPChromosome, processing_times,
                     sim: bool = False, uncertain_level: float = 0.1,
                     num_samples: int = 20,
                     rng: Optional[np.random.Generator] = None) -> float:
    """计算染色体在给定加工时间下的 makespan。

    chromosome: FJSSP 的 (OS, MS) 解
    processing_times: (J, M) 各操作的平均加工时间矩阵
    sim: 是否开启蒙特卡洛模拟。False 时直接用平均加工时间算一次；
         True 时每轮采样一个完整场景、调度一次，num_samples 轮后取 makespan 平均（论文 Algorithm 2）
    uncertain_level: 不确定性水平 UL（论文 Eq.12）
    num_samples: 蒙特卡洛轮数（默认 20，论文 Table 2 短模拟值）
    rng: 随机数生成器，默认 np.random.default_rng()
    """
    processing_times = np.asarray(processing_times, dtype=float)
    expected_shape = (chromosome.num_jobs, chromosome.num_machines)
    if processing_times.shape != expected_shape:
        raise ValueError(
            f"加工时间矩阵形状应为 {expected_shape}，实际为 {processing_times.shape}")
    if rng is None:
        rng = np.random.default_rng()

    ops = chromosome.decode()   # 只解码一次，蒙特卡洛各轮复用
    if not sim:
        return _schedule_makespan(ops, processing_times)

    total = 0.0
    for _ in range(num_samples):
        scenario = sample_scenario(processing_times, uncertain_level, rng)
        total += _schedule_makespan(ops, scenario)
    return total / num_samples


def compute_twte(chromosome: FJSSPChromosome, processing_times, priorities,
                 due_dates, sim: bool = False, uncertain_level: float = 0.1,
                 num_samples: int = 20,
                 rng: Optional[np.random.Generator] = None) -> float:
    """计算染色体在给定加工时间下的 TWTE（论文 Eq.1/3/4）。

    chromosome: FJSSP 的 (OS, MS) 解
    processing_times: (J, M) 各操作的平均加工时间矩阵
    priorities: (J,) 各作业优先级（拖延权重 = priority，提前权重 = 1/priority）
    due_dates: (J,) 各作业交期
    sim / uncertain_level / num_samples / rng: 含义与 compute_makespan 相同，
        蒙特卡洛模式为每轮采样一个完整场景、调度一次，n 轮取 TWTE 平均
    """
    processing_times = np.asarray(processing_times, dtype=float)
    priorities = np.asarray(priorities, dtype=float)
    due_dates = np.asarray(due_dates, dtype=float)
    if priorities.shape != (chromosome.num_jobs,) or due_dates.shape != (chromosome.num_jobs,):
        raise ValueError(
            f"priorities 与 due_dates 应为长度 {chromosome.num_jobs} 的一维数组，"
            f"实际为 {priorities.shape} 与 {due_dates.shape}")
    if np.any(priorities <= 0):
        raise ValueError("优先级必须为正数（提前权重为其倒数）")
    if rng is None:
        rng = np.random.default_rng()

    ops = chromosome.decode()   # 只解码一次，蒙特卡洛各轮复用
    if not sim:
        completions = _schedule_job_completions(ops, processing_times)
        return _twte_from_completions(completions, priorities, due_dates)

    total = 0.0
    for _ in range(num_samples):
        scenario = sample_scenario(processing_times, uncertain_level, rng)
        completions = _schedule_job_completions(ops, scenario)
        total += _twte_from_completions(completions, priorities, due_dates)
    return total / num_samples


def compute_objectives(chromosome: FJSSPChromosome, processing_times, priorities,
                       due_dates, sim: bool = False, uncertain_level: float = 0.1,
                       num_samples: int = 20,
                       rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """一次解码、每轮一次调度，同时返回 (makespan, TWTE)。

    蒙特卡洛模式下两个目标共享同一批场景采样（同一轮的场景既算 makespan 又算 TWTE），
    且每轮只调度一次，比分别调用 compute_makespan / compute_twte 节省一半调度开销。
    参数含义与 compute_twte 相同。
    """
    processing_times = np.asarray(processing_times, dtype=float)
    priorities = np.asarray(priorities, dtype=float)
    due_dates = np.asarray(due_dates, dtype=float)
    expected_shape = (chromosome.num_jobs, chromosome.num_machines)
    if processing_times.shape != expected_shape:
        raise ValueError(
            f"加工时间矩阵形状应为 {expected_shape}，实际为 {processing_times.shape}")
    if priorities.shape != (chromosome.num_jobs,) or due_dates.shape != (chromosome.num_jobs,):
        raise ValueError(
            f"priorities 与 due_dates 应为长度 {chromosome.num_jobs} 的一维数组，"
            f"实际为 {priorities.shape} 与 {due_dates.shape}")
    if np.any(priorities <= 0):
        raise ValueError("优先级必须为正数（提前权重为其倒数）")
    if rng is None:
        rng = np.random.default_rng()

    ops = chromosome.decode()   # 只解码一次，蒙特卡洛各轮复用
    if not sim:
        completions = _schedule_job_completions(ops, processing_times)
        return (float(completions.max()),
                _twte_from_completions(completions, priorities, due_dates))

    total_makespan, total_twte = 0.0, 0.0
    for _ in range(num_samples):
        scenario = sample_scenario(processing_times, uncertain_level, rng)
        completions = _schedule_job_completions(ops, scenario)
        total_makespan += completions.max()
        total_twte += _twte_from_completions(completions, priorities, due_dates)
    return total_makespan / num_samples, total_twte / num_samples


if __name__ == "__main__":
    # 手工可验例子：J=3, M=3，所有操作平均加工时间均为 1
    # 示例染色体 OS=(3,2,1,3,3,1,2,2,1), MS=(2,1,3,3,1,2,1,2,3)，
    # semi-active 调度手推：O31(0-1) O21(0-1) O11(0-1) O32(1-2) O33(2-3)
    #   O12(1-2) O22(2-3) O23(3-4) O13(3-4)，makespan = 4
    example = FJSSPChromosome(
        os=[3, 2, 1, 3, 3, 1, 2, 2, 1],
        ms=[2, 1, 3, 3, 1, 2, 1, 2, 3],
        num_machines=3,
    )
    mean_pt = np.ones((3, 3))

    deterministic = compute_makespan(example, mean_pt, sim=False)
    print("确定性 makespan =", deterministic, "(预期 4)")

    simulated = compute_makespan(
        example, mean_pt, sim=True, uncertain_level=0.1, num_samples=1000,
        rng=np.random.default_rng(42))
    print("模拟 makespan (UL=0.1, n=1000) =", round(simulated, 4),
          "(应略高于 4，对数正态场景下同一调度的期望 makespan >= 确定值)")

    # 10-5 实例全链路：随机粒子 -> 解码 -> 确定性/模拟 makespan
    from src.data_structures.particle import EOParticle
    chromosome = EOParticle.random(10, 5, rng=np.random.default_rng(7)).to_chromosome()
    pt = np.full((10, 5), 5.0)
    print("\n10-5 实例随机解:")
    print("确定性 makespan =", compute_makespan(chromosome, pt, sim=False))
    print("模拟 makespan (UL=0.25, n=10) =",
          round(compute_makespan(chromosome, pt, sim=True,
                                 uncertain_level=0.25, num_samples=20,
                                 rng=np.random.default_rng(7)), 4))

    # TWTE 手工可验例子：同上 3x3 例，完工时间 C1=4, C2=4, C3=3
    # 取 priority=(1,2,3), due_date=(3,4,5):
    #   J1: C=4>d=3, T=1 -> 1*1 = 1
    #   J2: C=4=d=4       -> 0
    #   J3: C=3<d=5, E=2 -> (1/3)*2 = 2/3
    #   TWTE = 1 + 0 + 2/3 = 5/3
    priorities = np.array([1.0, 2.0, 3.0])
    due_dates = np.array([3.0, 4.0, 5.0])
    twte = compute_twte(example, mean_pt, priorities, due_dates, sim=False)
    print("\n确定性 TWTE =", round(twte, 6), "(预期 5/3 =", round(5 / 3, 6), ")")
    print("模拟 TWTE (UL=0.1, n=1000) =",
          round(compute_twte(example, mean_pt, priorities, due_dates, sim=True,
                             uncertain_level=0.1, num_samples=1000,
                             rng=np.random.default_rng(42)), 4))

    # 双目标一次调度同时算出（NSGA-II 评估入口）
    m, t = compute_objectives(example, mean_pt, priorities, due_dates,
                              sim=True, uncertain_level=0.1, num_samples=1000,
                              rng=np.random.default_rng(42))
    print("\ncompute_objectives (UL=0.1, n=1000): makespan =", round(m, 4),
          ", TWTE =", round(t, 4))