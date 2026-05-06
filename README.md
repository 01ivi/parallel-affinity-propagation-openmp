# parallel-affinity-propagation-openmp

## 1. 项目简介

本项目是一个 C++17 + OpenMP 的并行 Affinity Propagation 聚类课程项目。相比 K-means，AP 聚类不需要预先指定簇数，而是通过样本之间的消息传递自动选择 exemplar 作为簇代表点。

原始 AP 需要维护完整的 `n x n` 相似度矩阵和消息矩阵，内存与计算开销较大。本项目的核心优化是:

| 设计 | 说明 |
| --- | --- |
| 稀疏近邻图 | 每个样本只保留 top-L 个近邻和自环，降低 AP 消息传递规模 |
| CSR/CSC 双索引 | 行更新 responsibility，列更新 availability |
| OpenMP 并行 | 并行构图、并行行最大值更新、并行列归约、并行标签分配 |
| damping 稳定迭代 | 使用阻尼抑制 AP 常见震荡问题 |

程序不依赖第三方库，只使用 C++ 标准库和 OpenMP。实验脚本会生成 `results/result.csv`，可直接用于课程报告中的运行时间、加速比、并行效率、簇数和目标函数分析。

## 2. AP 聚类原理

Affinity Propagation 是一种基于消息传递的无监督聚类算法。它把每个样本都视为潜在 exemplar，通过迭代更新两类消息来决定哪些样本最终成为簇中心。

两类消息为:

```text
r(i,k): responsibility，样本 i 选择 k 作为代表点的适合程度
a(i,k): availability，样本 k 作为 i 的代表点的可信程度
```

相似度定义为负平方欧氏距离:

```text
s(i,k) = -||x_i - x_k||^2
```

自相似度 `s(k,k)` 称为 preference，用来控制 exemplar 数量。preference 越大，最终簇数通常越多；preference 越小，最终簇数通常越少。

AP 的核心更新公式为:

```text
r(i,k) = s(i,k) - max_{k' != k} { a(i,k') + s(i,k') }

a(i,k) = min(0, r(k,k) + sum_{i' notin {i,k}} max(0, r(i',k)))  i != k
a(k,k) = sum_{i' != k} max(0, r(i',k))
```

当 `a(k,k) + r(k,k) > 0` 时，样本 `k` 被选为 exemplar。

## 3. 稀疏近邻图设计

原始 AP 对所有样本对 `(i,k)` 进行消息传递，复杂度较高:

```text
时间复杂度: O(I × n^2)
空间复杂度: O(n^2)
```

其中 `I` 为迭代次数，`n` 为样本数。

本项目先为每个样本计算 top-L 个最近邻，只在这些边上保留相似度和消息，并额外保留自环 `(i,i)`。这样稀疏 AP 的规模近似为:

```text
边数: n × (L + 1)
时间复杂度: O(I × n × L)
空间复杂度: O(n × L)
```

该设计保留了 AP 自动选簇数的特点，同时显著减少消息矩阵规模，适合并行计算实验。

## 4. OpenMP 并行设计

本项目的并行化主要围绕 AP 的行更新和列更新展开。

### 4.1 并行构造 top-L 近邻图

每个样本 `i` 独立计算到其他样本的距离，并选出 top-L 个近邻:

```cpp
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < n; ++i) {
    // compute distances from sample i to all other samples
    // keep top-L nearest neighbors
}
```

不同样本之间没有写冲突，适合按行并行。

### 4.2 responsibility 行并行

更新 `r(i,k)` 时，需要在同一行中计算:

```text
max_{k' != k} { a(i,k') + s(i,k') }
```

为了避免每个 `k` 都重复扫描整行，程序对每行只计算一次第一大值和第二大值。若当前边是第一大值所在边，则使用第二大值作为 `max_except`；否则使用第一大值。

该阶段按行并行:

```text
每个线程负责若干样本行，独立更新这些行上的 r 消息。
```

### 4.3 availability 列并行

更新 `a(i,k)` 时，需要按列统计:

```text
sum max(0, r(i',k))
```

因此程序除了 CSR 行索引外，还构建了 CSC-like 列索引:

```text
row_ptr, col_idx, sim      用于行更新
col_ptr, col_edges         用于列更新
```

列更新阶段按 exemplar 候选点 `k` 并行，每个线程负责若干列，先求正 responsibility 之和，再更新该列所有 availability 消息。

### 4.4 标签分配并行

AP 选出 exemplar 后，每个样本独立寻找最近 exemplar，该阶段按样本并行。

## 5. damping 与收敛

AP 容易出现震荡，因此每次消息更新都使用 damping:

```text
message = damping × old_message + (1 - damping) × new_message
```

程序用 `max_delta` 表示本轮最大消息变化量。若满足:

```text
max_delta < tol
```

或达到 `max_iter`，则停止迭代。

## 6. 复杂度分析

设样本数量为 `n`，维度为 `d`，每个样本保留近邻数为 `L`，迭代次数为 `I`，线程数为 `p`。

构图阶段需要计算样本两两距离:

```text
O(n^2 × d / p)
```

AP 消息迭代阶段:

```text
O(I × n × L / p)
```

空间复杂度:

```text
O(n × d + n × L)
```

相比稠密 AP 的 `O(n^2)` 消息矩阵，稀疏近邻图能明显降低内存压力。

## 7. 编译与运行

编译 OpenMP 版本:

```bash
make
```

Makefile 默认使用:

```text
-std=c++17 -O3 -march=native -fopenmp
```

如果本机没有 OpenMP，可以先编译串行验证版:

```bash
make serial
```

生成高斯簇数据并运行:

```bash
./parallel_ap --generate 3000 10 6 --neighbors 30 --preference -10 --threads 4 --output results/labels.csv
```

读取真实 CSV 数据:

```bash
./parallel_ap --input data/sample.csv --neighbors 40 --damping 0.75 --threads 8 --output results/labels.csv
```

程序会向标准输出打印一行 CSV 风格指标:

```text
mode,n,dim,neighbors,threads,build_time,ap_time,total_time,iterations,clusters,objective,edges,preference,damping,max_delta
```

## 8. 实验运行方法

运行自动实验脚本:

```bash
bash scripts/run_experiments.sh
```

脚本默认参数:

```text
n = 1000, 3000, 6000
dim = 10
true_k = 6
neighbors = 30
max_iter = 100
damping = 0.75
preference = -10
threads = 1, 2, 4, 8
```

可通过环境变量覆盖默认参数:

```bash
NS="1000 2000 4000" THREADS="1 2 4 8 16" NEIGHBORS=50 bash scripts/run_experiments.sh
```

结果保存到:

```text
results/result.csv
```

CSV 表头:

```text
n,dim,true_k,neighbors,mode,threads,build_time,ap_time,total_time,iterations,clusters,objective,edges,preference,damping,max_delta,speedup,efficiency
```

## 9. 实验结果表格模板

| n | dim | true_k | neighbors | mode | threads | total_time | iterations | clusters | objective | speedup | efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 10 | 6 | 30 | serial | 1 |  |  |  |  | 1.000000 | 1.000000 |
| 1000 | 10 | 6 | 30 | sparse_openmp | 1 |  |  |  |  |  |  |
| 1000 | 10 | 6 | 30 | sparse_openmp | 2 |  |  |  |  |  |  |
| 1000 | 10 | 6 | 30 | sparse_openmp | 4 |  |  |  |  |  |  |
| 1000 | 10 | 6 | 30 | sparse_openmp | 8 |  |  |  |  |  |  |

## 10. 结果分析写作建议

运行时间分析:

比较不同数据规模 `n` 和不同线程数下的 `total_time`。由于构图阶段需要计算两两距离，数据规模增大后并行化收益更明显。

加速比分析:

对每个 `n` 使用单线程时间作为基准:

```text
speedup = T_1 / T_p
```

并行效率分析:

```text
efficiency = speedup / threads
```

线程数增加后效率可能下降，原因包括线程调度开销、内存带宽竞争、列更新负载不均和 AP 每轮迭代中的同步点。

稀疏化效果分析:

重点分析 `neighbors=L` 对运行时间和聚类质量的影响。`L` 越小，边数越少，速度越快，但可能丢失部分相似关系；`L` 越大，聚类质量可能更稳定，但计算和内存开销增加。

preference 分析:

AP 的簇数由 preference 控制。可以固定其他参数，改变 `--preference` 或 `--preference-quantile`，观察最终 `clusters` 的变化。

## 11. 总结

本项目将原始 Affinity Propagation 的稠密消息传递改造成稀疏近邻图消息传递，并使用 OpenMP 对构图、responsibility 更新、availability 更新和标签分配进行并行化。该方案相比普通 K-means 更有算法特色: 不需要预设簇数，同时又能围绕 `O(n^2)` 瓶颈展开明确的并行优化和实验分析。
