# CLAUDE.md

# 项目说明

复现 Nessari et al. (2024) 的 **EO-Sim-NSGA-II**：EO 先以单目标（makespan）优化连续随机键种群，100 代后整群解码为离散 (OS, MS) 染色体作 NSGA-II 初始种群；NSGA-II 以 Monte Carlo 短模拟估计双目标 (E[makespan], E[TWTE]) 演化，最终 Pareto 成员长模拟重评价。基线 Sim-NSGA-II（随机初始化，其余全同）

## 参考文献

实现核心算法与数据结构时，必须以以下两篇文献为准：

### 1. @papers/A hybrid simheuristic algorithm for solving bi-objective stochastic flexible job shop scheduling problems.pdf
- 算法：SIM-NSGA-II, Monte Carlo Simulation
- 要求：严格按论文实现, 算法实现时只阅读第3、4节的内容

### 2. @papers/Equilibrium optimizer.pdf
#### 2.1 @papers/EO_pseudocode.md
- 算法：Equilibrium Optimizer
- 要求：严格按论文实现, 算法实现时只阅读第3节的内容以及Fig.3中的伪代码(存放在EO_pseudocode中)

## 编码要求
- 涉及到向量等数据结构时，使用numpy库，处理excel时使用pandas库
- 实现算法前，先阅读上述文献的相关章节
- 如果文献里没有明确说明，不要自行假设，先问我
