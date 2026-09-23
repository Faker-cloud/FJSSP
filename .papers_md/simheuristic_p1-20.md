# A hybrid simheuristic algorithm for solving bi-objective stochastic flexible job shop scheduling problems

<!-- image-->

Saman Nessari a, Reza Tavakkoli-Moghaddam a,b,∗, Hessam Bakhshi-Khaniki a, Ali Bozorgi-Amiri a

a School of Industrial Engineering, College of Engineering, University of Tehran, Tehran, Iran

b Research Center of Performance and Productivity Analysis, Istinye University, Istanbul, Turkey

## A R T I C L E I N F O

Dataset link: https://github.com/SamanNsr/Hy brid-Simheuristic-BiObj-Stoch-FJSSP

Keywords:   
Flexible job shop scheduling   
Multi-objective stochastic optimization   
Simheuristics   
Equilibrium optimizer   
Non-dominated sorting genetic algorithm

## A B S T R A C T

The flexible job shop scheduling problem (FJSSP) is a complex optimization challenge that plays a crucial role in enhancing productivity and efficiency in modern manufacturing systems, aimed at optimizing the allocation of jobs to a variable set of machines. This paper introduces an algorithm to tackle the FJSSP by minimizing makespan and total weighted earliness and tardiness under uncertainty. This hybrid algorithm effectively addresses the complexities of stochastic multi-objective optimization by integrating the equilibrium optimizer (EO) as an initial solutions generator, Non-dominated sorting genetic algorithm II (NSGA-II), and simulation techniques. The algorithm’s effectiveness is validated by showcasing specific instances and delivering decision results for optimal scheduling across varying levels of uncertainty. Results reveal the algorithm’s consistent superiority in managing the complexities of stochastic parameters across various problem scales, achieving lower makespan and improved Pareto front quality compared to existing methods. Particularly notable is the algorithm’s faster convergence and robust performance, as validated by the statistical Wilcoxon test, which confirms its reliability and efficacy in handling dynamic scheduling situations. These findings underscore the algorithm’s potential in providing flexible, robust solutions. The proposed algorithm’s unique balance of exploitative and explorative capabilities within a simulation framework enables effective handling of uncertainty in the FJSSP, offering flexibility and customization that is adaptable to various scheduling environments.

## 1. Introduction

Efficient production scheduling plays a crucial role in the strategic planning and management of contemporary manufacturing enterprises. A well-devised scheduling strategy can significantly enhance productivity and optimize resource utilization [1]. A flexible job shop scheduling problem (FJSSP) is an NP-hard classical scheduling problem [2], where machine flexibility, the ability of machines to perform various operations, plays a pivotal role. This flexibility allows for dynamic adjustment in task assignments, contributing to effective use of resources and enhanced responsiveness to job requirements. Numerous meta-heuristic algorithms, including non-dominated sorting genetic algorithm (NSGA II) [3], particle swarm optimization (PSO) [4], grey wolf optimizer (GWO) [5], and simulated annealing (SA) [6] have been introduced to address the challenges of the FJSSP. A job shop scheduling problem (JSSP) has been influenced by various uncertainties [7], with efforts to address these uncertainties evolving over time. Many uncertainties can arise due to factors such as varying processing times [8], machine breakdowns [9], unexpected arrival of new jobs [10], and changes in job priorities [11].

Equilibrium optimizer (EO), a physics-inspired algorithm, achieves optimal solutions through random updates of particle concentrations. Guided by equilibrium candidates, EO balances exploration and exploitation, effectively avoiding local optima. It outperforms popular meta-heuristic algorithms like PSO, GWO, and genetic algorithm (GA) in tests on mathematical and engineering problems, making it a concise and competitive optimization solution [12].

In summarizing the points made, the main motivation of this study is to develop a hybrid simheuristic algorithm combining the EO, NSGA-II, and Monte Carlo simulation to leverage their strengths to tackle the FJSSP. The critical importance of integrating these methods lies in addressing the dual objectives of makespan and total weighted earliness in the context of stochastic environments, which has not been concurrently considered in prior studies and is the focus of this research. Conversely, most related works have employed either purely deterministic or less integrative stochastic approaches, whereas this study utilizes a simulation to manage uncertainty dynamically, a method that considers a broader array of stochastic scenarios. The approach of conducting the study involves initially using EO to generate a highquality population by optimizing the single objective of makespan, which then sets a solid foundation for NSGA-II to handle the subsequent bi-objective optimization. The simulation aspect helps in managing the inherent uncertainty effectively. In this context, the fundamental innovations of this study are (1) the initial high-quality population generation using EO for more effective multi-objective optimization; (2) the novel integration of EO, NSGA-II, and simulation to effectively handle the stochastic nature of the FJSSP; (3) the use of a structured simheuristic approach to provide robust, adaptable scheduling solutions suitable for various manufacturing environments.

The structure of the paper is as follows. Section 2 presents the literature review. Sections 3 and 4 present the problem description and the solution method, respectively. Sections 5 and 6 present the findings and the managerial insight, respectively. Section 7 presents the conclusion.

## 2. Literature review

The FJSSP is an extension of the classical JSSP and is considered an NP-hard combinatorial optimization problem with significant realworld applications [2]. Due to its complexity and relevance, numerous solution approaches have been proposed in the literature, which can be broadly classified into three categories: exact algorithms, heuristics, and meta-heuristics [13]. The FJSSP has important real-world applications [14] in various industries, including manufacturing [15–17] printing [18], the automotive industry [19,20], auto parts remanufacturing [21], and the aerospace industry [22]. Researchers have developed specialized solution methods and models to address the specific requirements and constraints encountered in these practical scenarios. Boyer et al. [23] introduce the generalized FJSSP, inspired by seamless rolled ring manufacturing. It extends the classical FJSSP with constraints like machine capacity, time lags, holding times, and sequence-dependent setups. Jim et al. [24] introduce a model for the green FJSSP with time-of-use electricity pricing, aiming to minimize costs and carbon emissions and maximize customer satisfaction. It proposes an improved genetic algorithm (IGA) with a four-layer encoding scheme for chromosomes. This algorithm combines global and local search strategies for population initialization, separate crossover and mutation operations for different chromosome layers, and selects dominant individuals to enhance solution quality.

Researchers have formulated integer linear programming (ILP) and Mixed Integer Linear programming (MILP) models to solve the FJSSP optimally [25–28]. These models aim to optimize objectives such as makespan, tardiness, and workload balancing, subject to various constraints. Branch-and-bound techniques have also been employed in conjunction with ILP/MILP models to solve the FJSSP [29–31]. Furthermore, constraint programming (CP) has emerged as a promising exact approach for solving the highly complex FJSSP and its many variants. CP models have demonstrated the ability to optimally solve some medium-scale instances and provide feasible solutions for large practical cases that are challenging for other exact methods like Mixed Integer Programming (MIP) [13]. However, meta-heuristics still often outperform pure CP in terms of solution quality, especially in larger instances, within reasonable computational times [32]. As a result, hybrid CP meta-heuristic algorithms that combine the strengths of both paradigms are gaining interest [33]. While tuning CP models remains difficult due to the complex constraints, CP offers a valuable complement to meta-heuristic methods for tackling the FJSSP’s combinatorial complexity [34,35].

Various heuristic methods, including dispatching rules [36], beam search [37], and other constructive and improvement heuristics, have been developed to obtain good solutions for the FJSSP in reasonable computational times [27,38,39]. These heuristics often incorporate problem-specific knowledge and dispatching rules to guide the search process effectively. Moreover, Meta-heuristic approaches, which can be further divided into population-based and single-based methods, have been extensively explored for solving the FJSSP [32]. GAs are among the most widely applied population-based meta-heuristics for the FJSSP [40–42]. These algorithms employ various encoding schemes, crossover operators, and mutation operators to evolve a population of solutions. GAs have successfully been applied to both singleobjective and multi-objective variants of the FJSSP [17,43]. Other population-based techniques (e.g., PSO [44], ant colony optimization (ACO) [45], artificial bee colony (ABC) [46], and their variants) have also been proposed to solve the FJSSP and its extensions. In the context of single-based methods, SA algorithms, with different cooling schedules and neighborhood structures, have been widely used to tackle the FJSSP [47]. Tabu search (TS) and its variants (e.g., reactive TS) have also been successfully applied to the FJSSP [48]. Variable neighborhood search [49], iterated local search (ILS) [50], and their hybridizations with other meta-heuristics have been proposed to enhance the exploration and exploitation capabilities of the search process [40].

Recent studies have explored quantum annealing (QA) as a metaheuristic for the FJSSP. Schworm et al. [51] demonstrate the efficiency of a QA-based approach using quantum hardware for the FJSSP, outperforming classical methods and showcasing its potential for advanced manufacturing scheduling problems. However, previous QA applications to the FJSSP have been limited to small problems and single objectives, failing to capture the multi-criteria nature of real-world scheduling. Schworm et al. [52] filled that gap by proposing a novel QA-based solving algorithm that can optimize multiple objectives like makespan, workload, and job priorities simultaneously for the FJSSP.

Abu-Marrul et al. [53] introduced a simheuristic algorithm merging Iterated Greedy with Monte Carlo simulation to tackle a complex machine scheduling issue in the oil and gas sector, dealing with uncertainty in stochastic processing times and release dates. A robust JSSP dealing with machine unavailability from maintenance activities is tackled using simheuristic algorithms [54]. Caldeira and Gnanavelbabu [55] proposed a simheuristic to minimize the expected makespan in the stochastic FJSSP, outperforming meta-heuristics and offering a solution for addressing uncertainty. Li et al. [56] introduced a novel simheuristic algorithm for the FJSSP with uncertain processing times, outperforming other algorithms under uncertainty.

A bi-objective FJSSP is addressed, aiming to minimize earliness, tardiness, and risks from uncertain processing times using a simheuristic [57]. Castaneda et al. [58] presented a methodology combining meta-heuristic, simulation, and fuzzy methods to tackle stochastic and fuzzy uncertainty in a permutation flow shop scheduling problem (FSSP), outperforming regular optimization. Wang et al. [59] introduced the hybrid adaptive differential evolution algorithm for fuzzy multi-objective JSSP, demonstrating superiority in solution quality and stability. Gheisariha et al. [60] developed an enhanced multiobjective harmony search algorithm for a flexible FSSP. It addresses sequence-dependent setup times, transportation, and rework.

Lim et al. [61] presented an SA-based hyper-heuristic for solving the complex FJSSP, demonstrating the potential for automatically configuring heuristics in scheduling. Saqlain et al. [62] introduced a Monte Carlo tree search algorithm for the FJSSP, outperforming baselines in minimizing makespan. Three algorithms are proposed for parallel FSSP with stochastic times, offering insights into variability and performance tradeoffs [63]. A simheuristic algorithm is explored for the stochastically distributed assembly permutation FSSP, delivering solutions balancing makespan and risk [64]. Fu et al. [65] proposed a multi-objective ABC algorithm for a hybrid FSSP model with stochastic times. Experiments showed it outperforms other algorithms in optimizing quality and tardiness.

According to Table 1, the reviewed articles tackle various scheduling problems under uncertainty, including FJSSP and FSSP models, using simheuristics and multi-objective optimization algorithms. However, there is a gap in addressing the bi-objective stochastic FJSSP with the specific objectives of minimizing makespan and the total weighted tardiness and earliness.

Table 1 Literature review.
<table><tr><td>Study</td><td>Manufacturing system</td><td>Stochastic</td><td>Source of uncertainty</td><td>Objectives</td><td>Simheuristic</td><td>Method</td><td>Robustness measures</td></tr><tr><td>Boyer et al. (2021)</td><td>Generalized flexible job</td><td></td><td>None</td><td>Minimize completion time</td><td></td><td>MILP +Constraint Programming</td><td></td></tr><tr><td>Fu et al. (2021)</td><td>shop Hybrid flow shop</td><td>✓</td><td>Processing times</td><td>Maximize quality and minimize</td><td>✓</td><td>Multi-objective ABC +Stochastic</td><td></td></tr><tr><td>Caldeira and Gnanavel- babu</td><td>Flexible job shop</td><td>✓</td><td>Processing time</td><td>tardiness Minimize expected makespan</td><td>✓</td><td>simulation Jaya algorithm +Monte Carlo simulation</td><td>Reliability plots and boxplots</td></tr><tr><td>(2021) Li, Gong and Lu (2022)</td><td>Flexible job shop</td><td>✓</td><td>Processing times (fuzzy)</td><td>Minimize makespan and machine</td><td></td><td>Reinforcement learning</td><td></td></tr><tr><td>Wang, Gao and Pedrycz (2022)</td><td>Job shop</td><td>✓</td><td>Processing times and</td><td>workload Minimize completion time, delay time, and</td><td></td><td>Hybrid differential evolution</td><td></td></tr><tr><td>Castaneda et al. (2022)</td><td>Flow shop</td><td>✓</td><td>Processing</td><td>energy use Minimize expected makespan</td><td>✓</td><td>Meta-heuristic +Monte Carlo</td><td>Confidence intervals of survival function</td></tr><tr><td>Souza et al. (2022)</td><td>Job shop</td><td>✓</td><td>Machine availability</td><td>Minimize the weighted sum of expected makespan</td><td>✓</td><td>+Fuzzy Genetic algorithm +Simulation optimization</td><td>Quality and solution robustness</td></tr><tr><td>Rodríguez- Espinosa et al. (2023)</td><td>Flexible job shop</td><td>✓</td><td>Processing times</td><td>deviations Minimize earliness and tardiness and</td><td>✓</td><td>NSGA-II +Monte Carlo simulation</td><td>Deterioration minimization;</td></tr><tr><td>Abu-Marrul et al. (2023)</td><td>Parallel</td><td>✓</td><td>Processing times and</td><td>Minimize weighted</td><td>✓</td><td>Iterated Greedy +Monte Carlo</td><td>deterioration Confidence intervals; trade-off</td></tr><tr><td>Tutumlu and Saraç (2023)</td><td>Flexible job shop</td><td></td><td>None</td><td>Minimize makespan</td><td></td><td>Simulation MIP +Hybrid genetic algorithm</td><td>analysis; re-planning strategy</td></tr><tr><td>Saqlain, Ali and Lee (2023)</td><td>Flexible job shop</td><td>✓</td><td>Processing time</td><td>Minimize makespan and waiting times Maximize improve</td><td>✓</td><td>Reinforcement learning</td><td></td></tr><tr><td>Lim, Wong and Chin (2023)</td><td>Flexible job shop</td><td></td><td>None</td><td>utilization Minimize makespan</td><td></td><td>Neighbourhood structures</td><td></td></tr><tr><td>Schworm et al. (2023)</td><td>Flexible job shop</td><td></td><td>None</td><td>Minimize makespan, tardiness, energy</td><td></td><td>Quantum annealing</td><td></td></tr><tr><td>Schworm et al. (2024)</td><td>Flexible job shop</td><td></td><td>None</td><td>Maximize machine utilization Minimize makespan and total workload</td><td></td><td>Quantum annealing</td><td></td></tr><tr><td>Jia et al. (2024)</td><td>Green flexible job shop</td><td></td><td>None</td><td>Minimize cost (energy cost +order delay cost), carbon emissions</td><td></td><td>Improved genetic algorithm</td><td></td></tr><tr><td>This study</td><td>Flexible job shop</td><td>✓</td><td>Processing times</td><td>Maximize customer satisfaction Minimize makespan, and</td><td>✓</td><td>EO-NSGA-II +Monte Carlo</td><td>Wilcoxon test</td></tr></table>

The papers that come closest examine FJSSP models with makespan, earliness, and tardiness objectives [57], as well as a fuzzy FJSSP [61], but do not directly target the combined makespan and total earliness and tardiness objectives for a stochastic FJSSP. In addition, existing research lacks comprehensive approaches, particularly in the context of large-scale problem instances that leverage combining multiple metaheuristic approaches with simulation to address the uncertainty and this specific problem scenario.

To address this gap, the proposed hybrid simheuristic method puts forward an integration of the EO, NSGA-II, and Monte Carlo simulation. The key innovations include:

• Generating a high-quality initial population for the multi-objective algorithm in a hybrid simheuristic algorithm by first optimizing a single objective (i.e., makespan).

• Integrating this initial population into a simheuristic NSGA-II to then handle the bi-objectives of makespan and total weighted earliness, under uncertainty and large-scale problems.

• Considering stochastic processing time in the context of the FJSSP simheuristics.

• Using statistical Wilcoxon tests to assess the robustness of the algorithm across different scenarios.

In essence, our approach uniquely combines single and multi-objective optimization, leverages their strengths through a hybrid simheuristic framework, and targets the open challenge of a bi-objective stochastic FJSSP to advance the state-of-the-art. The robustness analysis further evaluates the consistency of results across different contexts.

## 3. Problem description

The FJSSP extends the classical JSSP by allowing each operation to be processed on a set of capable machines rather than a single designated machine. The FJSSP addresses the optimal arrangement of n jobs across m machines within a flexible manufacturing system, aiming to achieve objectives such as reducing makespan and managing job earliness and tardiness [66]. Each job i consists of a sequence of operation $O _ { i j }$ that must be processed sequentially, following specific precedence constraints. Notably, each operation $O _ { i j }$ within job i has a subset $M _ { i j }$ of m machines on which it can be processed. This means that for each operation, a decision must be made not only regarding the sequence of operations on each machine but also which specific machine from the available subset $M _ { i j }$ should be assigned to process that operation. The FJSSP introduces additional flexibility in the scheduling process by allowing operations to be assigned to different capable machines, potentially leading to better resource utilization and shorter makespans. However, this increased flexibility also makes the problem more complex to solve optimally, as the search space for possible solutions is significantly larger compared to the classical JSSP.

## 3.1. Problem formulation

The notations used in the proposed mathematical model are summarized as follows:

n Total number of jobs

m Total number of machines

$O _ { i j }$ Operation j of job i

$M _ { i j }$ Subsets of machines for $O _ { i j }$

$C _ { m a x }$ Makespan

TWTE Total weighted tardiness and earliness

$W T _ { i }$ Tardiness weight of job i

$W E _ { i }$ Earliness weight of job i

$T _ { i }$ Tardiness of job i

$E _ { i }$ Earliness of job i

$S t _ { i j }$ Starting time of processing of $O _ { i j }$

$P t _ { i j }$ Processing time of $O _ { i j }$

$C _ { i j }$ Completion time of $O _ { i j }$

$d _ { i }$ Due date of job i

$X _ { i j k }$ 1 if machine k is selected for operation $O _ { i j } ; 0 _ { \it { i } }$ , otherwise

$$
M i n T W T E = \sum _ { i } W T _ { i } T _ { i } + W E _ { i } E _ { i }\tag{1}
$$

?? ????????????

(2)

s.t.

$$
C _ { i j } - S t _ { i j } = \sum _ { k \in M _ { i j } } P t _ { i j } X _ { i j k } , \forall i , j
$$

$$
T _ { i } \ge m a x 0 , C _ { i j } - d _ { i } , \forall i , j\tag{3}
$$

$$
E _ { i } \ge m a x 0 , d _ { i } - C _ { i j } , \forall i , j\tag{4}
$$

$$
C _ { m a x } \geq S t _ { i j } + P t _ { i j } , \forall i , j\tag{5}
$$

(6)

$$
C _ { i j } - C _ { i , j - 1 } \geq P t _ { i j } X _ { i j k } , \forall i , j = ( 2 , \ldots , n ) , k\tag{7}
$$

$$
C _ { i j } \le S t _ { i , j + 1 } , \forall i , j = ( 1 , \ldots , n - 1 )\tag{8}
$$

$$
C _ { i j } - C _ { i ^ { \prime } , j ^ { \prime } } \geq P t _ { i j } X _ { i j k } X _ { i ^ { \prime } j ^ { \prime } k } , \forall i , j , i ^ { \prime } , j ^ { \prime } , k
$$

$$
\sum _ { k = 1 } ^ { M _ { i j } } X _ { i j k } = 1 , \forall i , j\tag{9}
$$

(10)

The FJSSP is formulated as a mathematical model with the objectives of minimizing the total weighted tardiness and earliness (1), as well as minimizing the makespan (2). The processing time $P t _ { i j }$ for each operation j of job i on machine m is defined based on the difference between the completion time (Cij) and the start time $( S t _ { i j } )$ (3). These completion times and machine assignments are decision variables.

The total earliness $E _ { i }$ and tardiness $T _ { i }$ for each job i with due date $d _ { i }$ are calculated based on the differences with the job completion time $C _ { i j } ,$ which is constrained to be after all its operations are complete. These feed into the weighted objective function for total earliness and tardiness (4), (5). The makespan $C _ { m a x }$ depends on the maximum $C _ { i j }$ (6). Precedence constraints are enforced between operations of each job based on start times $S t _ { i j } \ ( 7 )$ , (8). Non-overlap constraints prevent overlapping execution on the same machine m based on processing periods (9). Finally, machine assignment constraints allocate each operation j of job i to exactly one of its eligible machines $M _ { i j }$ (9).

In essence, the mathematical model aims to leverage the flexibilities in routing operations across machines to minimize weighted earliness and tardiness and makespan, while ensuring precedence satisfaction and non-overlap during schedule construction. The complexity warrants meta-heuristic solution approaches.

## 3.2. Stochastic processing times

In scheduling problems like the FJSSP, processing times are traditionally assumed to be deterministic, meaning they are known precisely. However, in actual manufacturing settings, processing times often exhibit variability due to factors such as machine breakdowns, operator skill levels, material availability, and unforeseen delays. Stochastic processing times introduce randomness or uncertainty into scheduling. Instead of fixed, predetermined processing times, these times are treated as random variables. This implies that the duration needed to complete an operation on a machine is not precisely predictable but follows a probability distribution. Modeling processing times as stochastic variables provides a more realistic portrayal of the dynamic nature of manufacturing processes. Common probability distributions used for this purpose include the normal, exponential, uniform, and lognormal distributions, among others. The selection of a distribution depends on the specific characteristics of the manufacturing process and the type of uncertainty involved.

The stochastic processing times $S P t _ { i j }$ are made to follow a lognormal distribution (11) with variance which UL denotes uncertainty level with three values of 0.1, 0.25, and 0.5 (12). The Log-Normal distribution is characterized by two parameters, ?? (13) and ?? (14), corresponding to each operation. The lognormal distribution is suitable for modeling positive random variables such as processing times. By defining the variance through the uncertainty level UL parameter, different scenarios are generated from low to high variability. The processing time for each operation then becomes a lognormal random variable based on its ?? and ?? sampled from the distributions corresponding to the UL.

Using this stochastic modeling approach, the processing times take on a range of values across simulations as opposed to fixed deterministic quantities. The FJSSP is executed repeatedly to evaluate the

objectives under these randomized processing time scenarios. The performance and robustness of scheduling solutions can then be analyzed under controlled settings of uncertainty. Employing variance as a driver offers flexibility in problem complexity and enables studying scheduler behavior and tradeoffs at different stochasticity levels.

<!-- image-->  
Fig. 1. General framework of the EO-sim-NSGA-II.

In effect, the integration of stochastic and simulation modules provides levers to systematically vary the degree of uncertainty in processing times. This facilitates a robustness assessment of scheduling solutions across different stochastic problem environments for the FJSSP.

$$
S P t _ { i j } = E [ S P t _ { i j } ]\tag{11}
$$

$$
V a r [ S P t _ { i j } ] = U L . E [ S P t _ { i j } ]\tag{12}
$$

$$
\mu _ { i j } = \ln E [ S P t _ { i j } ] - { \frac { 1 } { 2 } } \ln \left( 1 + { \frac { V a r [ S P t _ { i j } ] } { E [ S P t _ { i j } ] ^ { 2 } } } \right)
$$

$$
\sigma _ { i j } = \left| \sqrt { \ln \left( 1 + \frac { V a r [ S P t _ { i j } ] } { E [ S P t _ { i j } ] ^ { 2 } } \right) } \right|\tag{13}
$$

(14)

## 4. Methodology

An optimization framework is proposed that synergistically combines a hybrid meta-heuristic algorithm with a simheuristic approach. This framework leverages the explorative strengths of meta-heuristics and the probabilistic modeling capability of simulations to effectively tackle complex, real-world optimization problems. As shown in Fig. 1, the framework comprises two key components working in tandem.

## 4.1. Sim-NSGA-II

Simheuristic represents an innovative optimization approach that seamlessly integrates simulation and meta-heuristic methods. Instead of relying on a deterministic way, a simheuristic harnesses simulations to evaluate solutions, thereby adeptly accommodating the intricate and probabilistic nature inherent in real-world problem-solving.

As depicted in Algorithm 1, at the core of this framework, the NSGA-II algorithm orchestrates the exploration of solution spaces, employing its adaptive mechanisms to intelligently navigate through potential solutions. The NSGA-II, a widely adopted multi-objective evolutionary algorithm, utilizes techniques such as nondominated sorting, crowding distance, and elitist selection to uncover the Pareto frontier. It operates by maintaining a population of candidate solutions and iteratively evolving this population through genetic operations such as selection, crossover, and mutation. Central to its efficacy is the concept of Pareto dominance, wherein one solution is deemed superior to another if it excels in at least one objective without worsening any other objective.

Simultaneously, the Monte Carlo simulation injects a probabilistic element into the optimization process. It systematically generates and analyzes numerous scenarios by sampling from probability distributions, thereby generating diverse data points that comprehensively assess solution quality under varying conditions. As depicted in Algorithm 2, Monte Carlo simulation, which is a costly and computationally intensive technique, relies on random sampling to yield numerical results for complex problems fraught with uncertainty. By iterative sampling from probability distributions representing uncertain input parameters, Monte Carlo simulation furnishes estimates of the probability distribution of output variables, enabling decision-makers to evaluate the robustness and reliability of different solutions under uncertainty.

Algorithm 1 serves as a blueprint, outlining the procedural steps that illustrate the collaboration within the simheuristic framework. To execute Algorithm 1 effectively, the configuration of four key parameters is required, namely the population size for the meta-heuristic algorithm, the designated number of generations for evolutionary processes, as well as the mutation rate and crossover rate. The solutions of the deterministic model undergo a short simulation in a stochastic model, enabling the swift evaluation of its fitness concerning the defined objectives. This preliminary phase serves as a rapid screening process, swiftly gauging the potential of solutions within a constrained timeframe. Following this, the final Pareto frontier advances to the final step, undergoing an extensive long simulation, allowing for a deeper understanding of solution performance and robustness.

Algorithm 1: SIM-NSGA-II   
Input: Population size N, number of generations G, mutation rate   
mr, crossover rate cr   
Output: Pareto front approximation   
P ← InitializePopulation(N );   
ShortSimulation(pop);   
for g = 1 to G do   
Create an empty offspring population Q   
while |??| < ?? do   
Select parents from P using binary tournament selection   
Perform crossover and mutation to create offspring   
Add offspring to Q   
end while   
??′ ← ????????????(??)   
ShortSimulation(??′);   
Merge P and ??′ into a combined population R   
Perform non-dominated sorting on R   
Assign crowding distance to individuals in front of R   
Select the best N individuals from R based on crowding   
distance   
Update P with the selected individuals   
Increment generation counter g   
end for   
for s ∈ P do   
LongSimulation(s);   
end for   
Return Pareto front approximation from P

Algorithm 2: Monte Carlo Simulation   
Input: Number of simulations S, distribution D   
Output: Estimated value   
sum ← 0   
for s = 1 to S do   
Generate a random sample x from distribution D Calculate the   
value f (x)   
sum ← sum + f (x)   
end for   
average ← sum/S   
Return average

## 4.2. EO-sim-NSGA-II

A two-stage hybrid simheuristic algorithm is proposed for the resolution of the multi-objective optimization problem. In the first stage, the problem is reformulated into a single objective, employing the EO, as demonstrated in Algorithm 3. Initialization of a random solution population by EO initiates an evolution process driven by equilibrium forces of attraction and repulsion. This process steers the population towards diverse optima, reflecting trade-offs between exploration and exploitation.

Algorithm 3: InitializePopulation: Equilibrium Optimizer   
Input: Population size N, number of iterations T   
Output: Optimized solution   
Initialize Random population P with N individuals Evaluate fitness   
of individuals in P   
for t = 1 to T do   
Update the position of each individual in P   
Evaluate the fitness of individuals in P   
Sort individuals in P based on fitness   
Calculate the mean position of the top k individuals in P   
Calculate the standard deviation of the positions of the top k   
individuals in P   
for each individual p in P do   
Update the position of p using the Equilibrium Equation   
end for   
Evaluate the fitness of individuals in P   
Sort individuals in P based on fitness   
Update the equilibrium factor based on the fitness   
improvement   
Update the step size based on the fitness improvement   
Apply the boundary constraints to the positions of individuals   
in P   
end for   
Return the best individual in P

The set of equilibrated single-objective solutions produced by EO serves as the initial population for the NSGA-II in the second stage. By utilizing the EO solutions as a seed, sim-NSGA-II commences its search from pre-optimized starting points. This EO-Sim-NSGA-II framework leverages the single-objective exploration of EO in the initial stage to provide well-initialized starting points for the focused multi-objective search conducted by NSGA-II in the subsequent stage.

Moreover, in various real-world scenarios, multiple objectives are often interconnected or interdependent, implying that improvements in one objective can have positive or negative effects on other objectives. These relationships can vary across different regions or solutions within the search space. Consequently, while the overall objectives may not globally conflict, trade-offs and conflicts between objectives may arise at specific points or local optima within the problem space. Thereby, the diverse solutions generated by EO for one objective contribute to increased diversity within the Pareto front. Detailed implementation specifics of both algorithms are provided later. Performance evaluations against standalone applications on multi-objective benchmark functions validate the hybrid approach’s capacity to enhance convergence.

Sequence of job operations  
<!-- image-->  
Fig. 2. FJSSP representation coding.

## 4.3. FJSSP representation

In a scenario involving the FJSSP with J jobs and M machines, the coding representation is structured with J x M variables that assign sequence operations of each job to their eligible machines. Specifically, each of the J jobs contains multiple operations that will only be processed once on each machine.

As an illustration, Fig. 2 depicts an example with 3 jobs and 3 machines, it can be observed that this encoding denotes a permutation that sequences each job exactly once on each machine according to the job-machine sequence. By encapsulating complete scheduling decisions in a compact format, this representation facilitates convenient manipulation by the optimization algorithm to improve objectives. The vector is decoded into feasible schedules satisfying constraints.

## 5. Computational experiments

The computational experiments were conducted on a benchmark set consisting of 9 instances, each varying in dimension, by considering parameters for running the experiments from Table 2, to comprehensively assess the performance of the EO-SIM-NSGA-II algorithm. The implementation of the simheuristic was in Python programming language. These experiments were executed on a hardware configuration featuring an Apple Silicon M1 Pro chip with 16 GB of RAM, operating on macOS.

For each dataset instance, the EO-SIM-NSGA-II algorithm was executed, generating optimization results that showcase its efficacy in solving diverse problem dimensions. Subsequently, to provide a thorough analysis, convergence charts were constructed to illustrate the algorithm’s convergence behavior across various datasets. These charts offer insights into the convergence speed and stability of the algorithm across different problem instances. Moreover, to ascertain the robustness and statistical significance of the obtained results, a Wilcoxon test was conducted. The results obtained from these computational experiments serve to elucidate the strengths and efficacy of the EO-SIM-NSGA-II algorithm. The ensuing discussion will delve into the insights gleaned from these experiments, providing a comprehensive understanding of the algorithm’s performance, its strengths, and areas for potential improvement.

A key metric used in the comparative analysis is the percent gap $( \varDelta _ { \mathrm { o } } )$ in best objective values between the deterministic $( f _ { d } )$ and stochastic solutions $( f _ { s } ) _ { : }$ , quantified as Eq. (15). A positive gap percentage indicates the stochastic solution is worse than the deterministic case.

$$
\Delta _ { o } = \frac { f _ { s } - f _ { d } } { f _ { d } }\tag{15}
$$

Table 2  
Parameters configuration for running experiment.
<table><tr><td colspan="2">ment.</td></tr><tr><td>Parameters</td><td>Value</td></tr><tr><td>Population size</td><td>100</td></tr><tr><td>Crossover rate</td><td>0.8</td></tr><tr><td>Mutation rate</td><td>0.2</td></tr><tr><td>EO iteration</td><td>100</td></tr><tr><td>Total iteration</td><td>400</td></tr><tr><td>Long simulation iteration</td><td>10000</td></tr><tr><td>Short simulation iteration</td><td>20</td></tr><tr><td>Runs per experiment</td><td>30</td></tr></table>

## 5.1. Experiments with three uncertainty levels for the stochastic parameters

The first experiment compares the EO-Sim-NSGA-II against the Sim-NSGA-II using the deterministic NSGA-II across low, medium, and high uncertainty levels. The uncertainty is modeled by setting the variance parameter UL to 0.5, 1.0, and 2.0 for the three scenarios respectively. Each instance executes 30 times with the best outcome chosen for analysis. For each run, the algorithms take the obtained Pareto frontiers, consider the best value of each objective as the best result, and use the metrics as per Eq. (1). Also, the mean number of Pareto-optimal solutions (NPS) and mean CPU times are examined. Table 3 summarizes the comparative results grouped by job and machine combinations for the variance settings.

Table 3 clearly illustrates that, in most instances, the EO-Sim-NSGA-II outperformed the Sim-NSGA-II, exhibiting a narrower gap compared to the best deterministic objectives discovered by the NSGA-II. Remarkably, certain runs reveal negative gaps, indicating solutions exceeding the deterministic performance despite uncertainty. This underscores the efficacy of integrating EO and NSGA-II to balance exploration and exploitation. Moreover, the EO-Sim-NSGA-II consistently excelled in discovering a greater number of high-quality Pareto solutions, showcasing its proficiency in exploring diverse Pareto frontiers. Notably, the computational efficiency of the EO-Sim-NSGA-II, measured in CPU time for the same total epoch, surpassed that of the Sim-NSGA-II. This suggests that the EO-Sim-NSGA-II not only achieves superior solutions but also does so more swiftly and with reduced computational demands.

Also, Figs. 3a and 3b illustrate sample scheduling solutions of experiments E1 and E4, obtained by the EO-sim-NSGA-II algorithm, These Gantt charts offer valuable visualizations that highlight the algorithm’s ability to tackle problems. The inclusion of specific scheduling instances as figures provides examples that demonstrate the algorithm’s capabilities, enhancing the comprehension of its performance. Such targeted illustrations grounded in actual output data strengthen understanding and build further credibility in the algorithm’s operational performance within complex flexible manufacturing systems.

<!-- image-->  
(a) Experiment E1

<!-- image-->  
(b) Experiment E4

Fig. 3. Gantt chart of one of the solutions of two experiments of the EO-sim-NSGA-II.  
Results summary for each algorithm.
<table><tr><td></td><td></td><td></td><td colspan="5"></td><td colspan="4"></td></tr><tr><td>Uncertainty level</td><td>i</td><td>j-m</td><td>Sim-NSGA-II</td><td></td><td></td><td></td><td>EO-Sim-NSGA-II</td><td></td><td></td><td>NPS</td><td>CPU time</td></tr><tr><td>Low</td><td></td><td></td><td> $\Delta _ { o _ { 1 } }$ </td><td> $\varDelta _ { o _ { 2 } }$ </td><td>NPS</td><td>CPU time</td><td> $\Delta _ { o _ { 1 } }$ </td><td> $\varDelta _ { o _ { 2 } }$ </td><td></td><td></td><td></td></tr><tr><td></td><td>E1</td><td>10-5</td><td>0.072617</td><td>0.306785</td><td>10</td><td>18.02444</td><td>0.053791</td><td></td><td>0.158251</td><td>12</td><td>16.43649</td></tr><tr><td></td><td>E2</td><td>15-5</td><td>0.116044</td><td>0.185706</td><td>11</td><td>23.25542</td><td></td><td>0.050021</td><td>0.149527</td><td>16</td><td>20.40536</td></tr><tr><td></td><td>E3</td><td>20-5</td><td>0.085316</td><td>0.319801</td><td>15</td><td>28.21675</td><td></td><td>0.059555</td><td>0.213556</td><td>13</td><td>25.85585</td></tr><tr><td></td><td>E4</td><td>10-10</td><td>0.062024</td><td>0.06986</td><td>2</td><td>26.64939</td><td></td><td>0.069579</td><td>0.385338</td><td>5</td><td>22.41676</td></tr><tr><td></td><td>E5 E6</td><td>1510</td><td>0.112312 0.048216</td><td>0.242765</td><td>2</td><td>35.77363</td><td></td><td>0.025074 -0.02503</td><td>0.000431 -0.20761</td><td>12 4</td><td>31.90378 40.49214</td></tr><tr><td></td><td>E7</td><td>2010 30-10</td><td></td><td>0.040687</td><td>4</td><td>45.2828 64.99246</td><td></td><td>0.027714</td><td>-0.01672</td><td>6</td><td></td></tr><tr><td></td><td>E8</td><td>1515</td><td>0.135357 0.207685</td><td>0.145341 0.29455</td><td>2</td><td>48.57303</td><td></td><td>0.093616</td><td>0.00891</td><td>5</td><td>59.17984 40.60772</td></tr><tr><td></td><td>E9</td><td>2015</td><td>0.050795</td><td></td><td>2</td><td>62.10068</td><td></td><td>0.020341</td><td>-0.15765</td><td>7</td><td>56.48539</td></tr><tr><td>Medium</td><td>E10</td><td>10-5</td><td></td><td>0.11633</td><td>3</td><td>18.50975</td><td></td><td></td><td></td><td>3</td><td>15.05491</td></tr><tr><td></td><td>E11</td><td>15-5</td><td>0.505516 0.519203</td><td>0.381171 0.893717</td><td>1 7</td><td>23.81621</td><td>0.443966</td><td>0.471782</td><td>0.196241 0.666539</td><td>16</td><td>20.35887</td></tr><tr><td></td><td>E12</td><td>20-5</td><td>0.464871</td><td>0.921781</td><td>12</td><td>28.67281</td><td>0.421311</td><td></td><td>0.585453</td><td>9</td><td>25.76782</td></tr><tr><td></td><td>E13</td><td></td><td></td><td></td><td>6</td><td>26.83027</td><td>0.388874</td><td></td><td>1.394518</td><td>7</td><td></td></tr><tr><td></td><td></td><td>10-10</td><td>0.517445</td><td>1.397383</td><td></td><td></td><td></td><td></td><td></td><td></td><td>22.52793</td></tr></table>

(continued on next page)

Table 3 (continued).
<table><tr><td>Uncertainty level</td><td>i</td><td>j-m</td><td colspan="4">Sim-NSGA-II</td><td colspan="4">EO-Sim-NSGA-II</td></tr><tr><td rowspan="14">High</td><td>E14</td><td>15-10</td><td>0.459242</td><td>0.29012</td><td>2</td><td>36.09009</td><td>0.368225</td><td>0.253576</td><td>6</td><td>32.49778</td></tr><tr><td>E15</td><td>2010</td><td>0.501342</td><td>1.029642</td><td>1</td><td>45.68245</td><td>0.282518</td><td>0.562286</td><td>4</td><td>40.535</td></tr><tr><td>E16</td><td>30-10</td><td>0.561429</td><td>0.621553</td><td>2</td><td>65.78497</td><td>0.482571</td><td>0.469576</td><td>3</td><td>59.44553</td></tr><tr><td>E17</td><td>15-15</td><td>0.548119</td><td>0.226847</td><td>1</td><td>48.09727</td><td>0.535778</td><td>0.073819</td><td>3</td><td>40.63923</td></tr><tr><td>E18</td><td>20-15</td><td>0.471488</td><td>0.246217</td><td>2</td><td>61.80097</td><td>0.375723</td><td>0.128152</td><td>1</td><td>56.77659</td></tr><tr><td>E19</td><td>10-5</td><td>1.046676</td><td>1.190311</td><td>6</td><td>18.69963</td><td>0.89024</td><td>0.920006</td><td>7</td><td>14.733</td></tr><tr><td>E20</td><td>15-5</td><td>0.915215</td><td>0.895387</td><td>4</td><td>23.657481</td><td>0.891501</td><td>0.659131</td><td>15</td><td>20.28575</td></tr><tr><td>E21</td><td>20-5</td><td>0.942701</td><td>1.320942</td><td>7</td><td>28.5997</td><td>0.92256</td><td>1.104665</td><td>14</td><td>26.37614</td></tr><tr><td>E22</td><td>10-10</td><td>0.921795</td><td>1.583894</td><td>3</td><td>26.999842</td><td>0.906593</td><td>1.141633</td><td>9</td><td>22.37123</td></tr><tr><td>E23</td><td>15-10</td><td>0.91458</td><td>0.768772</td><td>1</td><td>35.862</td><td>0.858763</td><td>0.482745</td><td>2</td><td>30.98958</td></tr><tr><td>E24</td><td>20-10</td><td>0.937112</td><td>0.688554</td><td>9</td><td>46.17402</td><td>0.737709</td><td>0.385937</td><td>2</td><td>40.36287</td></tr><tr><td>E25</td><td>30-10</td><td>1.117048</td><td>1.083773</td><td>5</td><td>64.65103</td><td>1.036762</td><td>0.961248</td><td>7</td><td>59.73793</td></tr><tr><td>E26</td><td>1515</td><td>1.016709</td><td>1.276376</td><td>2</td><td>48.68396</td><td>1.009566</td><td>0.851142</td><td>2</td><td>40.76972</td></tr><tr><td>E27</td><td>20-15</td><td>1.003926</td><td>0.648607</td><td>2</td><td>62.8053</td><td>0.904132</td><td>0.483014</td><td>5</td><td>56.23009</td></tr></table>

<!-- image-->  
(a) Small instance run of E1

<!-- image-->  
(b) Medium instance run of E5

<!-- image-->  
(c) Large instance run of E9  
Fig. 4. Pareto frontier solutions of running in each experiment.

Also, Figs. 4a to 4c visually represent the Pareto frontiers across small, medium, and large dataset instances of experiments related to the EO-sim-NSGA-II algorithm, serving as illustrations to comprehend better the problem. The trade-off curves depicted offer a glimpse into the algorithm’s ability to balance the minimization of makespan and total earliness and tardiness for problems with different sizes.

Fig. 6 encapsulates the information from Table 3 in the form of distributions representing the gaps between the mean solutions found for two objectives across 30 runs for 27 experiments. This yields a total of 810 samples, comparing them with their corresponding deterministic values. These distributions are presented as boxplots, illustrating the gaps between sim-NSGA-II and EO-sim-NSGA-II across all benchmark experiments under varying levels of uncertainty. The superiority of the EO-sim-NSGA-II algorithm over the sim-NSGA-II algorithm is evident. Through this analysis, it is observed that the EOsim-NSGA-II variant outperforms not only the sim-NSGA-II algorithm but also the deterministic results. This is attributed to the exploration of deterministic initial solutions by the EO for the simheuristic algorithm, which sometimes results in negative gaps. Further substantiating the evaluations, convergence analyses were conducted across 30 runs for each of the 27 experiments, tracking the best global solution in each iteration. The mean makespan was computed per iteration by aggregating across runs, excluding the second objective, as EO focuses solely on makespan. Fig. 5 plots the convergence trends on 9 experiments with low-level uncertainty and different instance sizes, varying from small to large. The EO-Sim-NSGA-II displays faster convergence, achieving lower makespan faster across problem sizes. This highlights the power of hybridizing multi-objective search and rapid knowledge building. Furthermore, the greater gaps in large instance experiments showcase the effectiveness of hybridizing simheuristic over the simple simheuristic (i.e., sim-NSGA-II).

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 5. Convergence charts of experiments.

<!-- image-->

<!-- image-->

<!-- image-->  
Fig. 6. Boxplot analysis of uncertainty levels.

## 5.2. Robustness analysis

A Wilcoxon statistical test (5% significance level) validates the robustness and significance of the performance of the EO-Sim-NSGA-II’s gains over the Sim-NSGA-II across different experiments. By comparing the best objective values in Pareto solutions from 30 runs of each experiment, it evaluated algorithm consistency and reliability. The outcomes in Table 4 highlight scenarios with p-values < 0.05, indicating significant differences between the two algorithms. The positive sign indicates the superiority of EO-sim-NSGA-II over Sim-NSGA-II, while the negative sign indicates the vice versa. The equal sign indicates that there is no significant difference. The results reveal that the EO-Sim-NSGA-II’s solutions demonstrate statistically superior optimization over Sim-NSGA-II. This underscores EO-Sim-NSGA-II’s robustness in delivering high-quality solutions despite problem variations. The Wilcoxon test provides statistical validation of the EO-Sim-NSGA-II’s reliability advantages and aptness for complex optimization, reinforcing its competitiveness. In essence, comprehensive analytical assessments cement algorithm robustness, informing managerial selections for stochastic scheduling scenarios. Embedding such validations facilitates sound decisions when navigating the intricacies of unpredictable optimization landscapes.

Table 4  
p-values of the Wilcoxon test EO-sim-NSGA-II vs Sim-NSGA-II with 5% significance.
<table><tr><td>Uncertainty level</td><td>j-m</td><td>01- p-value</td><td> $R _ { 1 }$ </td><td>O2- p-value</td><td> $R _ { 2 }$ </td></tr><tr><td>Low</td><td>E1</td><td>3.45E-02</td><td>+</td><td>4.66E−03</td><td>+</td></tr><tr><td></td><td>E2</td><td>5.59E-05</td><td>+</td><td>3.74E-03</td><td>+</td></tr><tr><td></td><td>E3</td><td>2.77E-03</td><td>+</td><td>2.34E-02</td><td>+</td></tr><tr><td></td><td>E4</td><td>3.13E-04a</td><td></td><td>2.13E-01a</td><td>=</td></tr><tr><td></td><td>E5</td><td>1.58E-03</td><td>+</td><td>1.06E-02</td><td>+</td></tr><tr><td></td><td>E6</td><td>4.41E−05</td><td>+</td><td>4.60E-04</td><td>+</td></tr><tr><td></td><td>E7</td><td>1.82E-05</td><td>+</td><td>2.09E-04</td><td>+</td></tr><tr><td></td><td>E8</td><td>4.97E-05</td><td>+</td><td>5.05E-04</td><td>+</td></tr><tr><td></td><td>E9</td><td>1.46E-03</td><td>+</td><td>1.06E-02</td><td>+</td></tr><tr><td>Medium</td><td>E10</td><td>1.06E−05</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E11</td><td>6.15E-08</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E12</td><td>1.64E-07</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E13</td><td>3.73E-09</td><td>+</td><td>8.03E−02a</td><td>=</td></tr><tr><td></td><td>E14</td><td>2.61E−08</td><td>+</td><td>2.55E-07</td><td>+</td></tr><tr><td></td><td>E15</td><td>1.86E-09</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E16</td><td>1.99E-06</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E17</td><td>7.32E-02a</td><td>=</td><td>3.73E-09</td><td>+</td></tr><tr><td></td><td>E18</td><td>7.99E−06</td><td>+</td><td>2.37E-05</td><td>+</td></tr><tr><td>High</td><td>E19</td><td>2.21E-02</td><td>+</td><td>1.82E-05</td><td>+</td></tr><tr><td></td><td>E20</td><td>9.61E−02a</td><td>=</td><td>2.61E-08</td><td>+</td></tr><tr><td></td><td>E21</td><td>7.67E-02a</td><td>=</td><td>2.09E-04</td><td>+</td></tr><tr><td></td><td>E22</td><td>1.58E-03</td><td>+</td><td>3.15E-07</td><td>+</td></tr><tr><td></td><td>E23</td><td>5.59E-05</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E24</td><td>6.15E-08</td><td>+</td><td>1.86E-09</td><td>+</td></tr><tr><td></td><td>E25</td><td>1.23E-04</td><td>+</td><td>3.24E-06</td><td>+</td></tr><tr><td></td><td>E26</td><td>5.02E−02a</td><td>=</td><td>5.59E-09</td><td>+</td></tr><tr><td></td><td>E27</td><td>9.52E-04</td><td>+</td><td>4.18E-04</td><td>+</td></tr></table>

a The worst value.

## 6. Managerial insights

Managing a stochastic bi-objective FJSSP entails substantial complexity from uncertainties that hamper optimization. Balancing minimized makespan against reduced earliness and tardiness becomes exceedingly challenging under unpredictability. Effectively navigating this landscape necessitates managerial cognizance of inherent tradeoffs and tailored algorithm selection adept at handling stochasticity. Comprehensive analytical assessments, including convergence and statistical appraisals like the Wilcoxon test, inform insightful scheduling strategies that reliably attain optimality across stochastic scenarios.

Additionally, embracing stochasticity underscores the criticality of adaptive resilience. Developing scheduling policies not only flexible but adaptable to random variations empowers agile responses to unforeseen events. Embedding adaptability fortifies the system’s capability to maintain optimal performance despite volatility. This entails continuously improving job completion times and managing earliness and tardiness deviations. The essence of managerial insight involves leveraging analytical insights and adaptive policies to foster an environment driving robust optimization across stochasticity. By combining algorithmic and strategic competencies, managers can aptly balance production objectives with resilience to uncertainties.

## 7. Conclusion

This paper has introduced a pioneering EO-Sim-NSGA-II algorithm that synergizes the explorative prowess of meta-heuristics with the exploitative focus of simheuristics for addressing a stochastic multiobjective FJSSP. Through an intricate integration of the EO and NSGA-II embedded within a Monte Carlo simulation framework, this method balances single and multi-objective searches to offer a potent solution tackling the combined minimization of makespan and total weighted earliness and tardiness amid uncertainty. Extensive computational experiments and statistical testing based on controlled variance injection have substantiated the EO-Sim-NSGA-II’s consistent outperformance against benchmarks, cementing accuracy, reliability, and robustness. The algorithm has displayed sophisticated learning capabilities in navigating complexity from dynamism and stochasticity across problem scales. Convergence studies have reaffirmed the hybrid’s competencies in achieving high-quality solutions faster, validating the methodological innovations. Key contributions including the uniqueness of the hybrid technique, holistic uncertainty-based analysis, and multifaceted performance assessments significantly advance the domains of optimization, simulation, and manufacturing research. The integrated approach offers flexibility and customizability based on scheduling environments. In essence, this work has successfully engineered and demonstrated an elite, resilient solution approach for conquering the onerous challenges of unpredictable multi-objective optimization in next-generation smart factories. By scientifically harnessing algorithmic power, it delivers transformative capabilities for managing manufacturing complexity. The insights propel further opportunities in embedding intelligence within industrial systems for the navigation of multifaceted dynamics.

However, it is important to acknowledge certain limitations in our proposed approach. First of all, while the integration of hybrid simheuristics offers promise in addressing the bi-objective stochastic FJSSP, the effectiveness of the approach may vary depending on the specific characteristics and complexities of the scheduling instances, it needs to be tested in other problem domains. Moreover, the performance of the proposed method could be influenced by the choice of parameter settings and the quality of the initial population generated during the optimization process, so it needs to be studied in a way that can be tuned for different problems. Finally, the proposed algorithm’s applicability to real-world manufacturing environments may require additional adaptation to account for practical constraints and dynamics, which were not fully explored in this study. Future research efforts should thus focus on mitigating these limitations and conducting extensive empirical evaluations to validate the applicability and effectiveness of the proposed approach across diverse problem domains and scales.

## Declaration of competing interest

The authors of this research entitled ‘‘A Hybrid Simheuristic Algorithm for Solving Bi-Objective Stochastic Flexible Job Shop Scheduling Problems’’ certify that there is no any affiliation with or involvement in any organization or entity with financial interest (e.g., honoraria; educational grants; participation in speakers’ bureaus; membership, employment, consultancies, stock ownership, or other equity interest; and expert testimony or patent-licensing arrangements), or non-financial interest (e.g., personal or professional relationships, affiliations, knowledge or beliefs) in the subject matter or materials discussed in this manuscript.

## Availability of data and material

The data and code that support the findings of this study are available at https://github.com/SamanNsr/Hybrid-Simheuristic-BiObj-Stoch-FJSSP.

## References

[1] J. Xie, X. Li, L. Gao, L. Gui, A new neighbourhood structure for job shop scheduling problems, Int. J. Prod. Res. 61 (2023) 2147–2161, http://dx.doi.org/ 10.1080/00207543.2022.2060772.

[2] J. Xie, L. Gao, K. Peng, X. Li, H. Li, Review on flexible job shop scheduling, IET Collab. Intell. Manuf. 1 (2019) 67–77, http://dx.doi.org/10.1049/iet-cim.2018. 0009.

[3] M. Rabiee, M. Zandieh, P. Ramezani, Bi-objective partial flexible job shop scheduling problem: NSGA-II, NRGA, MOGA and PAES approaches, Int. J. Prod. Res. 50 (2012) 7327–7342, http://dx.doi.org/10.1080/00207543.2011.648280.

[4] M. Nouiri, A. Bekrar, A. Jemai, S. Niar, A.C. Ammari, An effective and distributed particle swarm optimization algorithm for flexible job-shop scheduling problem, J. Intell. Manuf. 29 (2018) 603–615, http://dx.doi.org/10.1007/s10845-015- 1039-3.

[5] C. Lin, Z. Cao, M. Zhou, Learning-based grey wolf optimizer for stochastic flexible job shop scheduling, IEEE Trans. Autom. Sci. Eng. 19 (2022) 3659–3671, http://dx.doi.org/10.1109/TASE.2021.3129439.

[6] M.A. Cruz-Chávez, M.G. Martínez-Rangel, M.H. Cruz-Rosales, Accelerated simulated annealing algorithm applied to the flexible job shop scheduling problem, Int. Trans. Oper. Res. 24 (2017) 1119–1137, http://dx.doi.org/10.1111/itor. 12195.

[7] P.B. Luh, Chen Dong, L.S. Thakur, An effective approach for job-shop scheduling with uncertain processing requirements, IEEE Trans. Robot. Autom. 15 (1999) 328–339, http://dx.doi.org/10.1109/70.760354.

[8] Lei De-Ming, Xiong He-Jing, Job shop scheduling with stochastic processing time through genetic algorithm, in: 2008 International Conference on Machine Learning and Cybernetics, IEEE, 2008, pp. 941–946, http://dx.doi.org/10.1109/ ICMLC.2008.4620540.

[9] E. Ahmadi, M. Zandieh, M. Farrokh, S.M. Emami, A multi objective optimization approach for flexible job shop scheduling problem under random machine breakdown by evolutionary algorithms, Comput. Oper. Res. 73 (2016) 56–66, http://dx.doi.org/10.1016/j.cor.2016.03.009.

[10] D. Rahmani, M. Heydari, Robust and stable flow shop scheduling with unexpected arrivals of new jobs and uncertain processing times, J. Manuf. Syst. 33 (2014) 84–92, http://dx.doi.org/10.1016/j.jmsy.2013.03.004.

[11] D. Golenko-Ginzburg, S. Kesler, Z. Landsman, Industrial job-shop scheduling with random operations and different priorities, Int. J. Prod. Econ. 40 (1995) 185–195, http://dx.doi.org/10.1016/0925-5273(95)00078-8.

[12] A. Faramarzi, M. Heidarinejad, B. Stephens, S. Mirjalili, Equilibrium optimizer: A novel optimization algorithm, Knowl. Based Syst. 191 (2020) 105190, http: //dx.doi.org/10.1016/j.knosys.2019.105190.

[13] S. Dauzère-Pérès, J. Ding, L. Shen, K. Tamssaouet, The flexible job shop scheduling problem: A review, European J. Oper. Res. 314 (2024) 409–432, http://dx.doi.org/10.1016/J.EJOR.2023.05.017.

[14] C. Destouet, H. Tlahig, B. Bettayeb, B. Mazari, Flexible job shop scheduling problem under Industry 5.0: A survey on human reintegration, environmental consideration and resilience improvement, J. Manuf. Syst. 67 (2023) 155–173, http://dx.doi.org/10.1016/J.JMSY.2023.01.004.

[15] L.M. Steinbacher, D. Rippel, P. Schulze, A.K. Rohde, M. Freitag, Quality-based scheduling for a flexible job shop, J. Manuf. Syst. 70 (2023) 202–216, http: //dx.doi.org/10.1016/J.JMSY.2023.07.005.

[16] Q. Gong, J. Li, Z. Jiang, Y. Wang, A hierarchical integration scheduling method for flexible job shop with green lot splitting, Eng. Appl. Artif. Intell. 129 (2024) 107595, http://dx.doi.org/10.1016/J.ENGAPPAI.2023.107595.

[17] H. Tang, Y. Xiao, W. Zhang, D. Lei, J. Wang, T. Xu, A DQL-NSGA-III algorithm for solving the flexible job shop dynamic scheduling problem, Expert Syst. Appl. 237 (2024) 121723, http://dx.doi.org/10.1016/J.ESWA.2023.121723.

[18] W.T. Lunardi, E.G. Birgin, D.P. Ronconi, H. Voos, Metaheuristics for the online printing shop scheduling problem, European J. Oper. Res. 293 (2021) 419–441, http://dx.doi.org/10.1016/J.EJOR.2020.12.021.

[19] Q. Gao, F. Gu, L. Li, J. Guo, A framework of cloud–edge collaborated digital twin for flexible job shop scheduling with conflict-free routing, Robot. Comput. Integr. Manuf. 86 (2024) 102672, http://dx.doi.org/10.1016/J.RCIM.2023.102672.

[20] M.M. Wocker, F.F. Ostermeier, T. Wanninger, R. Zwinkau, J. Deuse, Flexible job shop scheduling with preventive maintenance consideration, J. Intell. Manuf. (2023) 1–23, http://dx.doi.org/10.1007/S10845-023-02114-3/TABLES/3.

[21] W. Zhang, Y. Zheng, R. Ahmad, An energy-efficient multi-objective integrated process planning and scheduling for a flexible job-shop-type remanufacturing system, Adv. Eng. Inform. 56 (2023) 102010, http://dx.doi.org/10.1016/J.AEI. 2023.102010.

[22] Z. Tian, X. Jiang, W. Liu, Z. Li, Dynamic energy-efficient scheduling of multivariety and small batch flexible job-shop: A case study for the aerospace industry, Comput. Ind. Eng. 178 (2023) 109111, http://dx.doi.org/10.1016/J.CIE.2023. 109111.

[23] V. Boyer, J. Vallikavungal, X. Cantú Rodríguez, M.A. Salazar-Aguilar, The generalized flexible job shop scheduling problem, Comput. Ind. Eng. 160 (2021) 107542, http://dx.doi.org/10.1016/J.CIE.2021.107542.

[24] S. Jia, Y. Yang, S. Li, S. Wang, A. Li, W. Cai, et al., The green flexible job-shop scheduling problem considering cost, carbon emissions, and customer satisfaction under time-of-use electricity pricing, Sustainability 16 (2024) 2443, http://dx.doi.org/10.3390/SU16062443.

[25] L. Meng, B. Zhang, K. Gao, P. Duan, An MILP model for energy-conscious flexible job shop problem with transportation and sequence-dependent setup times, Sustainability 15 (2022) 776, http://dx.doi.org/10.3390/SU15010776.

[26] J. Tang, G. Gong, N. Peng, K. Zhu, D. Huang, Q. Luo, An effective memetic algorithm for distributed flexible job shop scheduling problem considering integrated sequencing flexibility, Expert Syst. Appl. 242 (2024) 122734, http: //dx.doi.org/10.1016/J.ESWA.2023.122734.

[27] B. Ji, S. Zhang, S.S. Yu, B. Zhang, Mathematical modeling and a novel heuristic method for flexible job-shop batch scheduling problem with incompatible jobs, Sustainability 15 (2023) 1954, http://dx.doi.org/10.3390/SU15031954.

[28] G.A. Kasapidis, S. Dauzère-Pérès, D.C. Paraskevopoulos, P.P. Repoussis, C.D. Tarantilis, On the Multiresource Flexible Job-Shop Scheduling Problem with Arbitrary Precedence Graphs, vol. 32, 2023, pp. 2322–2330, http://dx.doi.org/ 10.1111/POMS.13977.

[29] A. Ahmadi-Javid, M. Haghi, P. Hooshangi-Tabrizi, Integrated job-shop scheduling in an FMS with heterogeneous transporters: MILP formulation, constraint programming, and branch-and-bound, Int. J. Prod. Res. (2023) http://dx.doi.org/ 10.1080/00207543.2023.2230489.

[30] J. Ahn, H.J. Kim, A branch and bound algorithm for scheduling of flexible manufacturing systems, IEEE Trans. Autom. Sci. Eng. (2023) http://dx.doi.org/ 10.1109/TASE.2023.3296087.

[31] C. Juvin, L. Houssin, P. Lopez, Logic-based benders decomposition for the preemptive flexible job-shop scheduling problem, Comput. Oper. Res. 152 (2023) 106156, http://dx.doi.org/10.1016/J.COR.2023.106156.

[32] M. Schlenkrich, S.N. Parragh, Solving large scale industrial production scheduling problems with complex constraints: an overview of the state-of-the-art, Procedia Comput. Sci. 217 (2023) 1028–1037, http://dx.doi.org/10.1016/J.PROCS.2022. 12.301.

[33] G. Ziadlou, S. Emami, E. Asadi-Gangraj, Network configuration distributed production scheduling problem: A constraint programming approach, Comput. Ind. Eng. 188 (2024) 109916, http://dx.doi.org/10.1016/J.CIE.2024.109916.

[34] D. Müller, D. Kress, Filter-and-fan approaches for scheduling flexible job shops under workforce constraints, Int. J. Prod. Res. 60 (2022) 4743–4765, http: //dx.doi.org/10.1080/00207543.2021.1937745.

[35] V. Boyer, J. Vallikavungal, X. Cantú Rodríguez, M.A. Salazar-Aguilar, The generalized flexible job shop scheduling problem, Comput. Ind. Eng. 160 (2021) 107542, http://dx.doi.org/10.1016/J.CIE.2021.107542.

[36] M. Thenarasu, K. Rameshkumar, M. Di Mascolo, S.P. Anbuudayasankar, Multicriteria scheduling of realistic flexible job shop: a novel approach for integrating simulation modelling and multi-criteria decision making, Int. J. Prod. Res. 62 (2024) 336–358, http://dx.doi.org/10.1080/00207543.2023.2238084.

[37] K. Hadj Salem, V. Jost, Y. Kieffer, L. Libralesso, S. Mancini, Minimizing makespan under data prefetching constraints for embedded vision systems: a study of optimization methods and their performance, Oper. Res. 22 (2022) 1639–1673, http://dx.doi.org/10.1007/S12351-021-00647-0/METRICS.

[38] K. Huang, W. Gong, C. Lu, An enhanced memetic algorithm with hierarchical heuristic neighborhood search for type-2 green fuzzy flexible job shop scheduling, Eng. Appl. Artif. Intell. 130 (2024) 107762, http://dx.doi.org/10.1016/J. ENGAPPAI.2023.107762.

[39] K.C.W. Lim, L.P. Wong, J.F. Chin, Hyper-heuristic for flexible job shop scheduling problem with stochastic job arrivals, Manuf. Lett. 36 (2023) 5–8, http://dx.doi. org/10.1016/J.MFGLET.2022.12.009.

[40] B. Tutumlu, T. Saraç, A MIP model and a hybrid genetic algorithm for flexible job-shop scheduling problem with job-splitting, Comput. Oper. Res. 155 (2023) 106222, http://dx.doi.org/10.1016/J.COR.2023.106222.

[41] L. Meng, W. Cheng, B. Zhang, W. Zou, W. Fang, P. Duan, An improved genetic algorithm for solving the multi-AGV flexible job shop scheduling problem, Sensors 23 (2023) 3815, http://dx.doi.org/10.3390/S23083815.

[42] M. Liu, J. Lv, S. Du, Y. Deng, X. Shen, Y. Zhou, Multi-resource constrained flexible job shop scheduling problem with fixture-pallet combinatorial optimisation, Comput. Ind. Eng. 188 (2024) 109903, http://dx.doi.org/10.1016/J.CIE.2024. 109903.

[43] Y. Tian, Z. Gao, L. Zhang, Y. Chen, T. Wang, A multi-objective optimization method for flexible job shop scheduling considering cutting-tool degradation with energy-saving measures, Mathematics 11 (2023) 324, http://dx.doi.org/10.3390/ MATH11020324.

[44] J. Shi, M. Chen, Y. Ma, F. Qiao, A new boredom-aware dual-resource constrained flexible job shop scheduling problem using a two-stage multi-objective particle swarm optimization algorithm, Inf. Sci. (N. Y.) 643 (2023) 119141, http://dx. doi.org/10.1016/J.INS.2023.119141.

[45] S. Yan, G. Zhang, J. Sun, W. Zhang, S. Yan, G. Zhang, et al., An improved ant colony optimization for solving the flexible job shop scheduling problem with multiple time constraints, Math. Biosci. Eng. 20 (2023) 7519–7547, http: //dx.doi.org/10.3934/MBE.2023325.

[46] Y. Li, C. Liao, L. Wang, Y. Xiao, Y. Cao, S. Guo, A reinforcement learningartificial bee colony algorithm for flexible job-shop scheduling problem with lot streaming, Appl. Soft. Comput. 146 (2023) 110658, http://dx.doi.org/10.1016/ J.ASOC.2023.110658.

[47] Z. Zhang, Y. Fu, K. Gao, H. Zhang, L. Wang, A cooperative evolutionary algorithm with simulated annealing for integrated scheduling of distributed flexible job shops and distribution, Swarm Evol. Comput. 85 (2024) 101467, http://dx.doi.org/10.1016/J.SWEVO.2023.101467.

[48] J. Xie, X. Li, L. Gao, L. Gui, A hybrid genetic tabu search algorithm for distributed flexible job shop scheduling problems, J. Manuf. Syst. 71 (2023) 82–94, http://dx.doi.org/10.1016/J.JMSY.2023.09.002.

[49] K. Sun, D. Zheng, H. Song, Z. Cheng, X. Lang, W. Yuan, et al., Hybrid genetic algorithm with variable neighborhood search for flexible job shop scheduling problem in a machining system, Expert Syst. Appl. 215 (2023) 119359, http: //dx.doi.org/10.1016/J.ESWA.2022.119359.

[50] W. Shao, Z. Shao, D. Pi, Lot sizing and scheduling problem in distributed heterogeneous hybrid flow shop and learning-driven iterated local search algorithm, IEEE Trans. Autom. Sci. Eng. (2023) http://dx.doi.org/10.1109/TASE. 2023.3326301.

[51] P. Schworm, X. Wu, M. Glatt, J.C. Aurich, Solving flexible job shop scheduling problems in manufacturing with Quantum Annealing, Prod. Eng. 17 (2023) 105–115, http://dx.doi.org/10.1007/S11740-022-01145-8/TABLES/6.

[52] P. Schworm, X. Wu, M. Klar, M. Glatt, J.C. Aurich, Multi-objective Quantum Annealing approach for solving flexible job shop scheduling in manufacturing, J. Manuf. Syst. 72 (2024) 142–153, http://dx.doi.org/10.1016/J.JMSY.2023.11. 015.

[53] V. Abu-Marrul, R. Martinelli, S. Hamacher, I. Gribkovskaia, Simheuristic algorithm for a stochastic parallel machine scheduling problem with periodic re-planning assessment, Ann. Oper. Res. 320 (2023) 547–572, http://dx.doi.org/ 10.1007/s10479-022-04534-5.

[54] R.L.C. Souza, A. Ghasemi, A. Saif, A. Gharaei, Robust job-shop scheduling under deterministic and stochastic unavailability constraints due to preventive and corrective maintenance, Comput. Ind. Eng. 168 (2022) 108130, http://dx.doi. org/10.1016/j.cie.2022.108130.

[55] R.H. Caldeira, A. Gnanavelbabu, A simheuristic approach for the flexible job shop scheduling problem with stochastic processing times, Simulation 97 (2021) 215–236, http://dx.doi.org/10.1177/0037549720968891.

[56] R. Li, W. Gong, C. Lu, A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job shop scheduling, Expert Syst. Appl. 203 (2022) 117380, http://dx.doi.org/10.1016/j.eswa.2022.117380.

[57] C.A. Rodríguez-Espinosa, E.M. González-Neira, G.M. Zambrano-Rey, A simheuristic approach using the NSGA-II to solve a bi-objective stochastic flexible job shop problem, J. Simul. (2023) 1–25, http://dx.doi.org/10.1080/17477778.2023. 2231877.

[58] J. Castaneda, X. Martin, M. Ammouriova, J. Panadero, A. Juan, A fuzzy simheuristic for the permutation flow shop problem under stochastic and fuzzy uncertainty, Mathematics 10 (2022) 1760, http://dx.doi.org/10.3390/ math10101760.

[59] G.-G. Wang, D. Gao, W. Pedrycz, Solving multiobjective fuzzy job-shop scheduling problem by a hybrid adaptive differential evolution algorithm, IEEE Trans. Ind. Inform. 18 (2022) 8519–8528, http://dx.doi.org/10.1109/TII.2022. 3165636.

[60] E. Gheisariha, M. Tavana, F. Jolai, M. Rabiee, A simulation–optimization model for solving flexible flow shop scheduling problems with rework and transportation, Math. Comput. Simulation 180 (2021) 152–178, http://dx.doi.org/10.1016/ J.MATCOM.2020.08.019.

[61] K.C.W. Lim, L.-P. Wong, J.F. Chin, Simulated-annealing-based hyper-heuristic for flexible job-shop scheduling, Eng. Optim. 55 (2023) 1635–1651, http://dx.doi. org/10.1080/0305215X.2022.2106477.

[62] M. Saqlain, S. Ali, J.Y. Lee, A Monte-Carlo tree search algorithm for the flexible job-shop scheduling in manufacturing systems, Flex. Serv. Manuf. J. 35 (2023) 548–571, http://dx.doi.org/10.1007/s10696-021-09437-4.

[63] E.M. Gonzalez-Neira, D. Ferone, S. Hatami, A.A. Juan, A biased-randomized simheuristic for the distributed assembly permutation flowshop problem with stochastic processing times, Simul. Model. Pract. Theory 79 (2017) 23–36, http://dx.doi.org/10.1016/j.simpat.2017.09.001.

[64] S. Hatami, L. Calvet, V. Fernández-Viagas, J.M. Framiñán, A.A. Juan, A simheuristic algorithm to set up starting times in the stochastic parallel flowshop problem, Simul. Model. Pract. Theory 86 (2018) 55–71, http://dx.doi.org/10. 1016/j.simpat.2018.04.005.

[65] Y. Fu, H. Wang, J. Wang, X. Pu, Multiobjective modeling and optimization for scheduling a stochastic hybrid flow shop with maximizing processing quality and minimizing total Tardiness, IEEE Syst. J. 15 (2021) 4696–4707, http://dx.doi. org/10.1109/JSYST.2020.3014093.

[66] Y. Zhou, J.J. Yang, L.Y. Zheng, Hyper-heuristic coevolution of machine assignment and job sequencing rules for multi-objective dynamic flexible job shop scheduling, IEEE Access 7 (2019) 68–88, http://dx.doi.org/10.1109/ACCESS. 2018.2883802.