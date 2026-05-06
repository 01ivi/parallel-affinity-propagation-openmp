# results

实验脚本会在这里生成:

- `result.csv`: 运行时间、加速比、并行效率、迭代次数、簇数等汇总指标。
- `labels_*.csv`: 每次运行得到的样本标签和 exemplar。

CSV 结果文件默认被 `.gitignore` 忽略，避免把大规模实验输出提交到仓库。
