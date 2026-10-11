# 学习与环境记录

按根 README 的阅读路线逐步添加笔记，建议顺序：

1. GEMM 与 Marlin 接口：维度、量化语义、调用链。
2. 权重打包与 scale 布局：用小矩阵手动追踪索引。
3. Warp 与 Tensor Core：`ldmatrix`、fragment、`mma.sync`。
4. 数据搬运与流水线：buffer、阶段、同步位置。
5. Striped partition 与归约：block 分工、workspace、输出。
6. 5090 Laptop 环境记录与首轮 profiling 分析。

每篇记录对应的源码提交和函数名，区分源码事实、自己的推导与待验证假设。环境文件记录实际安装版本和验证命令，验证前不称为可复现环境。
