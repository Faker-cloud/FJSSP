"""NSGA-II 阶段：Sim-NSGA-II 的多目标演化主体。

现已确定/实现：
- 交叉：OS 层 POX + MS 层段粒度交叉（crossover.py）。
- 变异：OS 随机两位交换 + MS 单作业块内互换，复合变异率 mr（mutation.py）。
- 选择（selection.py）：父代二元锦标赛 + 环境选择 select_best（直接消费前沿列表），读取染色体的 rank / crowding 属性。
- 非支配排序 + 拥挤距离（sorting.py）：fast_non_dominated_sort 赋 rank，crowding_distance_assignment 赋 crowding。

- 主循环（algorithm.py）：SimNSGAII —— 初始化（随机/EO 可切换）、sim 开关、无修复，串起以上算子。

染色体已挂载瞬时衍生状态：makespan、twte、rank、crowding。
"""