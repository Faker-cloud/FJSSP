# EO 算法伪代码（论文 Fig.3）

来源：`papers/img.png`（Equilibrium optimizer 论文 Fig.3 截图，经 OCR 提取并轻度规范化）。

```
Initialize the particle's populations i = 1, ..., n
Assign equilibrium candidates' fitness a large number
Assign free parameters a1 = 2; a2 = 1; GP = 0.5

While Iter < Max_iter
    For i = 1 : number of particles (n)
        Calculate fitness of i'th particle
        If fit(C_i) < fit(C_eq1)
            Replace C_eq1 with C_i, and fit(C_eq1) with fit(C_i)
        Elseif fit(C_i) > fit(C_eq1) & fit(C_i) < fit(C_eq2)
            Replace C_eq2 with C_i, and fit(C_eq2) with fit(C_i)
        Elseif fit(C_i) > fit(C_eq1) & fit(C_i) > fit(C_eq2) & fit(C_i) < fit(C_eq3)
            Replace C_eq3 with C_i, and fit(C_eq3) with fit(C_i)
        Elseif fit(C_i) > fit(C_eq1) & fit(C_i) > fit(C_eq2) & fit(C_i) > fit(C_eq3) & fit(C_i) < fit(C_eq4)
            Replace C_eq4 with C_i, and fit(C_eq4) with fit(C_i)
        End (If)
    End (For)

    C_ave = (C_eq1 + C_eq2 + C_eq3 + C_eq4) / 4
    Construct the equilibrium pool
        C_eq,pool = { C_eq(1), C_eq(2), C_eq(3), C_eq(4), C_eq(ave) }
    Accomplish memory saving (if Iter > 1)

    Assign  t = (1 - Iter/Max_iter)^(a2 * Iter/Max_iter)            Eq (9)

    For i = 1 : number of particles (n)
        Randomly choose one candidate from the equilibrium pool (vector)
        Generate random vectors λ, r
        Construct  F = a1 * sign(r - 0.5) * [e^(-λ t) - 1]           Eq (11)
        Construct  GCP = { 0.5*r1   if r2 >= GP
                         { 0        if r2 <  GP }                    Eq (15)
        Construct  G0 = GCP * (C_eq - λ * C)                        Eq (14)
        Construct  G = G0 * F                                       Eq (13)
        Update concentrations
            C = C_eq + (C - C_eq) * F + (G / (λ * V)) * (1 - F)     Eq (16)
    End (For)

    Iter = Iter + 1
End while
```

## 与论文正文的公式对应

| 行内记号 | 论文公式 | 说明 |
|---|---|---|
| `t` | Eq (9) | 时间因子，随迭代递减 |
| `F` | Eq (11) | 指数项（`a1 * sign(r-0.5) * [e^(-λt) - 1]`） |
| `GCP` | Eq (15) | 生成率控制参数（r2 >= GP 时 0.5*r1，否则 0） |
| `G0` | Eq (14) | `GCP * (C_eq - λ*C)` |
| `G` | Eq (13) | `G0 * F` |
| 浓度更新 | Eq (16) | `C = C_eq + (C - C_eq)*F + G/(λV)*(1 - F)` |

参数默认值（论文 3.1.3/3.1.4 节）：a1 = 2，a2 = 1，GP = 0.5；λ、r、r1、r2 为 [0,1] 均匀随机向量/数；V 视为单位。