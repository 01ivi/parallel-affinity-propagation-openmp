# data

这里可以放真实数据集 CSV 文件。程序要求每一行是一个样本，列为数值特征，例如:

```text
1.2,3.4,5.6
1.1,3.5,5.8
```

运行真实数据:

```bash
./parallel_ap --input data/sample.csv --neighbors 40 --threads 8 --output results/labels.csv
```

如果没有真实数据，也可以直接使用 `--generate N D TRUE_K` 生成高斯簇数据。
