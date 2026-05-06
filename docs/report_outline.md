# 并行 Affinity Propagation 聚类课程报告提纲

## 1. 绪论

介绍聚类问题背景、K-means 需要预先指定簇数的问题，以及 Affinity Propagation 能够通过消息传递自动发现 exemplar 的特点。说明 AP 的主要瓶颈是 `O(n^2)` 相似度矩阵和消息矩阵，因此适合做稀疏化与并行优化。

## 2. Affinity Propagation 算法原理

说明 AP 维护两类消息:

```text
r(i,k): responsibility，样本 i 选择 k 作为代表点的适合程度
a(i,k): availability，样本 k 作为 i 的代表点的可信程度
```

更新公式:

```text
r(i,k) = s(i,k) - max_{k' != k} { a(i,k') + s(i,k') }

a(i,k) = min(0, r(k,k) + sum_{i' notin {i,k}} max(0, r(i',k)))  i != k
a(k,k) = sum_{i' != k} max(0, r(i',k))
```

其中 `s(i,k) = -||x_i - x_k||^2`，`s(k,k)` 为 preference，控制最终簇数。

## 3. 稀疏近邻图优化

原始 AP 使用完整 `n x n` 相似度矩阵，空间复杂度和每轮计算复杂度均为 `O(n^2)`。本项目为每个样本只保留 top-L 个近邻，并额外保留自环 `(i,i)`，只在稀疏边上进行消息传递。

可分析:

```text
稠密 AP: O(I × n^2)
稀疏 AP: O(I × n × L)
```

其中 `I` 为迭代次数，`L` 为每个样本保留的近邻数。

## 4. OpenMP 并行设计

说明本项目的并行点:

1. 构造 top-L 近邻图时，按样本行并行。
2. 更新 `r(i,k)` 时，按行并行，每行计算第一大和第二大值。
3. 更新 `a(i,k)` 时，按列并行，每列求正 responsibility 之和。
4. 最终标签分配时，按样本并行。

## 5. 数据结构设计

说明稀疏图同时保存 CSR 和 CSC 两种索引:

```text
CSR: row_ptr, col_idx, sim
CSC-like: col_ptr, col_edges
```

CSR 适合 responsibility 的行更新，CSC-like 索引适合 availability 的列求和，避免在列更新时扫描所有边。

## 6. damping 与收敛

说明 AP 容易震荡，因此使用阻尼更新:

```text
message = damping × old_message + (1 - damping) × new_message
```

当最大消息变化 `max_delta < tol` 或达到 `max_iter` 时停止。

## 7. 实验环境

记录 CPU、内存、操作系统、g++ 版本、OpenMP 支持和编译参数:

```text
-std=c++17 -O3 -march=native -fopenmp
```

## 8. 实验参数

建议参数:

| 参数 | 取值 |
| --- | --- |
| n | 1000, 3000, 6000 |
| dim | 10 |
| true_k | 6 |
| neighbors | 30 |
| max_iter | 100 |
| damping | 0.75 |
| preference | -10 |
| threads | 1, 2, 4, 8 |

## 9. 结果指标

实验脚本输出:

```text
n,dim,true_k,neighbors,mode,threads,build_time,ap_time,total_time,iterations,clusters,objective,edges,preference,damping,max_delta,speedup,efficiency
```

可重点分析:

- `total_time`: 总运行时间。
- `speedup`: 相对单线程基准的加速比。
- `efficiency`: 并行效率。
- `clusters`: AP 自动得到的簇数。
- `edges`: 稀疏图边数，约为 `n × (L + 1)`。
- `objective`: 样本到 exemplar 的平方距离和。

## 10. 结果分析建议

运行时间分析:

比较不同 `n` 和不同线程数下的 `total_time`。数据规模较小时线程调度开销占比更高，数据规模增大后并行效果更明显。

加速比分析:

使用单线程 sparse AP 作为基准:

```text
speedup = T_1 / T_p
```

并行效率:

```text
efficiency = speedup / p
```

可解释效率下降原因: top-L 构图中的内存访问、列更新阶段的负载不均、内存带宽竞争以及 AP 迭代中的同步点。

## 11. 总结与展望

总结 AP 相比 K-means 的优势: 不需要预设簇数。总结本项目的优化点: 稀疏近邻图降低复杂度，CSR/CSC 双索引适配 AP 两类消息更新，OpenMP 行/列并行提高计算效率。

展望可加入真实数据集、近似近邻搜索、MPI 分布式稀疏 AP、CUDA 行最大值与列归约实现。
