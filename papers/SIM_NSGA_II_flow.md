# SIM-NSGA-II 算法流程

来源：论文「A hybrid simheuristic algorithm for solving bi-objective stochastic flexible job shop scheduling problems」第 4 节（Algorithm 1 / Algorithm 2 / Algorithm 3）、第 3 节问题描述与 4.3 节 FJSSP 编码。仅覆盖第 3、4 节内容。

本文件对应 EO 阶段的 `EO_pseudocode.md`，是 SIM-NSGA-II 阶段实现前的流程参照。

## 1. 总框架（论文 Fig.1，两阶段）

- **阶段一 EO**（已实现，`src/eo/equilibrium_optimizer.py`）：以连续随机键种群单目标最小化 makespan，运行 `Max_iter`（项目约定 100）代后，把**整群**粒子解码为 (OS, MS) 染色体。
- **阶段二 Sim-NSGA-II**：以阶段一解码出的种群作为 NSGA-II 的**初始种群**；以 Monte Carlo 短模拟估计双目标 (E[makespan], E[TWTE]) 演化；最终 Pareto 前沿成员用**长模拟**重新评价。

基线 **Sim-NSGA-II** 与 EO-Sim-NSGA-II 的唯一差别是初始种群：基线直接随机初始化，其余流程完全相同。

两阶段算法都在此处汇总：

| 算法 | 作用 | 位置 |
|---|---|---|
| Algorithm 3 `InitializePopulation: Equilibrium Optimizer` | 阶段一初始化种群（EO） | §4.2 |
| Algorithm 1 `SIM-NSGA-II` | 阶段二多目标演化的主循环 | §4.1 |
| Algorithm 2 `Monte Carlo Simulation` | 短/长模拟的评估原语 | §4.1 |

## 2. Algorithm 2 — Monte Carlo 模拟（评估原语）

短模拟（short simulation）与长模拟（long simulation）共用同一原语，差别仅在模拟次数 `S`。

```
Algorithm 2: Monte Carlo Simulation
Input: 模拟次数 S, 加工时间分布 D
Output: 目标估计值
    sum ← 0
    for s = 1 to S do
        从分布 D 采样一个完整加工时间场景 x
        计算该场景下的一次调度目标值 f(x)
        sum ← sum + f(x)
    end for
    average ← sum / S
    return average
```

对一个解：makespan、TWTE 各自按上述规则对 `S` 次场景取平均，得到 (E[makespan], E[TWTE]) 两个目标。随机加工时间分布见 §3.2（对数正态，式 11–14），由 `src/problem/fjssp_evaluation.py` 的 `sample_scenario` 实现。

## 3. Algorithm 1 — SIM-NSGA-II 主循环

```
Algorithm 1: SIM-NSGA-II
Input: 种群规模 N, 代数 G, 交叉率 cr, 变异率 mr
Output: Pareto 前沿近似
    P ← InitializePopulation(N)              # 基线：随机初始化；EO 版：整群解码自 EO
    ShortSimulation(P)                       # 短 MC 评估 P 的双目标
    for g = 1 to G do
        Q ← ∅                                # 空子代种群
        while |Q| < N do
            从 P 用二元锦标赛选择父母 (p1, p2)
            对父母做交叉(cr)与变异(mr)生成子代 (o1, o2)
            Q ← Q ∪ {o1, o2}
        end while
        ShortSimulation(Q)                   # 短 MC 评估子代
        R ← P ∪ Q                            # 合并成规模 2N 的种群
        对 R 做非支配排序 → 前沿 F1, F2, ...
        对各前沿内个体分配拥挤距离
        按 (前沿等级, 拥挤距离) 从 R 选择最优 N 个个体
        P ← 选中的 N 个个体
    end for
    for s ∈ P do
        LongSimulation(s)                    # 最终 Pareto 成员长 MC 重评价
    end for
    return P 的 Pareto 前沿近似
```

> 注：原文 Algorithm 1 的 OCR 有若干处 `??` 缺失（对应 `Q`、`N` 等符号），此处按语义补全，不改变算法结构。

## 4. 与现有代码的映射

| 流程环节 | 现状 | 位置 |
|---|---|---|
| 解结构 (OS, MS)，MS 为每作业机器置换 | ✅ | `src/problem/fjssp_chromosome.py` |
| 双目标评估 + MC 模拟开关 | ✅ | `src/problem/fjssp_evaluation.py::compute_objectives` |
| EO 整群解码作初始种群 | ✅ | `src/eo/equilibrium_optimizer.py::chromosomes` |
| 基线随机初始化 | ✅ | `FJSSPChromosome.random` |
| 二元锦标赛选择 | ✅ | `src/nsga2/selection.py` |
| 交叉算子（POX + MS 段交叉） | ✅ | `src/nsga2/crossover.py` |
| 变异算子（OS 两位交换 + MS 块内互换） | ✅ | `src/nsga2/mutation.py` |
| 非支配排序 / 拥挤距离 | ❌ 待实现 | — |
| 短/长模拟串联 | ❌ 待实现 | — |

## 5. 遗传算子 — 交叉、变异与选择（已确定）

每次交叉产 **2 个子代**，实现于 `src/nsga2/crossover.py`（入口 `crossover(parent1, parent2, rng)`）。

### 5.1 OS 层 — POX（Precedence Operation Crossover）
1. 随机作业子集 JA（其余作业为 JB），每作业独立二分（空集/全集时重采样，避免退化为复制父代）。
2. 子代1：把 P1 中作业号 ∈JA 的位置（含位置本身）照抄；空位按 P2 去掉 JA 后剩下的 JB 作业子序列顺序依次填入。
3. 子代2：对称交换继承方向（JA 位置取 P2，空位按 P1 的 JB 顺序填）。

示例（JA={J1}）：P1 OS=(1,2,2,3,1,3,2,1,3)，P2 OS=(2,3,1,1,3,2,3,1,2)
- 子代1：第 1,5,8 位照抄 1；空位第 2,3,4,6,7,9 位按 P2 去 J1 得 (2,3,3,2,3,2) 顺序填 → (1,2,3,3,1,2,3,1,2)
- 子代2：第 3,4,8 位照抄 1；空位按 P1 去 J1 得 (2,2,3,3,2,3) 顺序填 → (2,2,1,1,3,3,2,1,3)

### 5.2 MS 层 — 段粒度交叉
每个作业独立掷硬币选父代，整段搬运该作业的 M 台机器号；子代2 取子代1 的互补侧（子代1 取 P1 的作业、子代2 取 P2，反之亦然）。

示例（J1←P2, J2←P1, J3←P2）：MS₁=(3,1,2 | 1,3,2 | 2,1,3)，MS₂=(2,3,1 | 3,2,1 | 1,3,2)
- 子代1 = (2,3,1 | 1,3,2 | 1,3,2)
- 子代2 = (3,1,2 | 3,2,1 | 2,1,3)

两条规则都保持子代合法：POX 保持作业多重集不变（每作业仍出现 M 次）；MS 整段搬运的是合法父代中 1..M 的机器排列。

### 5.3 变异（复合变异，概率 mr）
每个子代以概率 mr 整体触发一次变异；触发时同时做两步：
1. OS 变异——随机两位交换：随机选两个不同位置，交换二者的作业号（例：第 5、6 位的 1 与 2 交换）。
2. MS 变异——单作业块内互换：随机抽一个作业，在其 M 台机器段内随机选两个不同位置交换（例：J3 段 (1,3,2)→(3,1,2)）。

两条变异都保持子代合法：OS 交换保持作业多重集不变；MS 块内互换仍为 1..M 的排列。

### 5.4 选择 — 二元锦标赛（已实现）
随机抽两个不同个体，按字典序 `(rank 升序、crowding 降序)` 取胜者；rank 相同看拥挤距离，拥挤大者胜。

实现于 `src/nsga2/selection.py`：
- `tournament_select(population, num_matches, rng)`：读取个体 `.rank`/`.crowding` 属性，摊平成 numpy 数组后向量化批量锦标赛，返回胜者对象。
- `select_parents(population, num_pairs, rng)`：**先 `rng.permutation` 随机打乱种群顺序**（消除非支配排序/环境选择留下的结构化顺序对平局打破的影响），再做 `2*num_pairs` 场独立锦标赛、两两配对，供 `crossover()` 使用。

时序约定：选择只在种群完成非支配排序 + 拥挤距离分配后调用（每个体的 `.rank`/`.crowding` 已赋真值；初始种群首代选择前也需先排序 + 算拥挤一次）。`rank`/`crowding`/`makespan`/`twte` 均作为 `FJSSPChromosome` 的瞬时属性存放。

## 6. 遗留待定项（实现前需再确认）

- **模拟次数**：短模拟 `S_short`（被删除的旧代码用 10，当前 `fjssp_evaluation` 默认 20）、长模拟 `S_long`，均待确认。
- **目标方向**：两目标均为最小化。旧代码目标序为 `[twte, makespan]`，现 `compute_objectives` 返回 `(makespan, twte)`，仅影响标签，不影响逻辑。
- **`Machines Sequence` 表**：数据集中的机器序列表**不用作固定指派**（已确认 MS 作为决策变量参与进化）。