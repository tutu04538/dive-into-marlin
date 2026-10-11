# Dive into Marlin

从基本的 tiled GEMM 出发，逐步读懂原始 Marlin，在 **RTX 5090 Laptop GPU** 上建立可复现的基线，再基于测量结果优化，后续在 **RTX 5090 Desktop** 上验证和调整。

本项目以算子学习和实验为主。当前已引入原始源码并规划学习、基线和实验目录；**尚未验证 5090 上的编译、正确性或性能**。

## 1. 学习对象与目标

这里的 Marlin 指 [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)，即面向 LLM 推理的 FP16 × INT4 矩阵乘法实现。学习对象是作者原始仓库，而非推理框架中后续演进的版本。

可以先用已有的 GEMM 知识理解它：

```text
A[M, K]：FP16 激活
Q[K, N]：逻辑上的 4-bit 权重，实际以重排后的 INT32 数据保存
S      ：量化缩放系数
B̂      ：由 Q 和 S 反量化得到的权重
C[M, N] = A × B̂
```

INT4 是权重的存储格式。内核在计算过程中将权重反量化为 FP16，再通过 Tensor Core 执行乘加；理解它需要把数据布局、数据搬运、反量化和计算调度串起来。正确性应对照同一份量化权重反量化后的 GEMM，避免把量化误差和内核误差混在一起。

近期目标是能解释一个 tile 从全局内存到输出的完整过程，并获得可信的 Laptop 基线。原作者在其他 GPU 上的加速比不作为本项目在 5090 上的预期结果。

## 2. 目录组织

```text
dive-into-marlin/
├── README.md                 # 项目入口、学习路线和实验约定
├── third_party/
│   ├── README.md             # 上游来源、固定版本与更新约定
│   └── marlin/               # 官方源码，Git submodule，保持原样
├── docs/                     # 阅读笔记、数据布局推导、环境记录
├── benchmarks/               # 后续添加的基线驱动与测量工具
├── experiments/              # 兼容性补丁、优化实现与实验说明
└── results/                  # 可提交的实验报告；raw/ 保存本地原始数据
```

除上游源码外，各目录目前只有用途说明。后续新增工具和实验时再扩展，避免预先搭建复杂框架。

## 3. 获取项目与原始源码

克隆本项目时使用 `git clone --recurse-submodules <本项目仓库地址>`。如果已经普通克隆，在项目根目录执行：

```bash
git submodule update --init --recursive
git submodule status
git -C third_party/marlin rev-parse HEAD
```

当前固定的上游提交：[`1f25790bdd49fba53106164a24666dade68d7c90`](https://github.com/IST-DASLab/marlin/commit/1f25790bdd49fba53106164a24666dade68d7c90)。父仓库会记录这个提交，普通初始化不会自动跟随上游最新版本。

阅读和建立基线时保持 submodule 干净。后续修改放到 `experiments/` 的补丁或独立实现中，记录其基于哪个上游提交；不要把只存在于本机的 submodule 提交作为依赖推送。

## 4. 从 tiled GEMM 到 Marlin 的阅读路线

按下面的顺序阅读，每一步先画图、解释数据流，再追求逐行理解。

| 阶段 | 阅读入口 | 需要回答的问题 / 产出 |
| --- | --- | --- |
| 0. 补齐基础 | 自己已有的 tiled GEMM | 画出 thread block、warp、thread 与 tile 的对应关系；解释合并访存、共享内存 bank conflict 和同步 |
| 1. 明确接口 | [`marlin/__init__.py`](third_party/marlin/marlin/__init__.py)、[`test.py`](third_party/marlin/test.py) | `M/N/K`、groupsize、workspace 分别是什么？先追踪 `Layer.forward → mul` |
| 2. 理解权重格式 | `Layer.pack`、`_get_perms` | 追踪一个 16×16 权重 tile：量化值如何重排、8 个 INT4 如何打包成 INT32、scale 如何对应 |
| 3. 理解计算单元 | [`marlin_cuda_kernel.cu`](third_party/marlin/marlin/marlin_cuda_kernel.cu) 中的 `ldsm4`、`dequant`、`mma` | 补齐 `ldmatrix`、寄存器 fragment 和 `mma.sync`；画出一个 warp 的数据布局 |
| 4. 理解流水线 | 同文件的 `cp_async*`、`fetch_to_shared`、`fetch_to_registers`、`matmul` | global → shared → registers 如何推进？哪个 buffer 正在写，哪个正在读，等待发生在哪里？ |
| 5. 理解任务划分 | `Marlin` 内核的 striped partition、`thread_block_reduce`、`global_reduce` | 多个 block 如何分配 K 方向的工作？为什么需要 workspace 和全局归约？ |
| 6. 连起调用链 | [`marlin_cuda.cpp`](third_party/marlin/marlin/marlin_cuda.cpp)、内核文件中的 `marlin_cuda` | Python 参数如何影响 tile、并行度和内核实例选择？ |
| 7. 阅读测量代码 | [`bench.py`](third_party/marlin/bench.py) | 测量包含什么？warmup、同步、数据和 shape 如何影响结果？ |

笔记放到 [`docs/`](docs/README.md)。每篇保留“我的理解、源码依据、仍未解答的问题”，方便后续用实验修正。

## 5. 在 5090 Laptop 上建立基线

### 5.1 先确定环境，再编译

记录 GPU 完整型号、compute capability、SM 数、显存，以及操作系统、驱动、CUDA Toolkit、PyTorch 与编译器版本。Laptop 还需要记录供电状态、性能模式、功耗限制、温度和时钟状态。

NVIDIA 将 GeForce RTX 5090 列为 compute capability 12.0，参见 [CUDA GPU 列表](https://developer.nvidia.com/cuda/gpus)。实际 Laptop 设备仍以本机查询为准；不要套用 Desktop 的 SM 数、显存或功耗参数。

原仓库写出的 CUDA / PyTorch 最低版本不是 5090 的环境保证。先选择支持目标 GPU 的 PyTorch CUDA 构建和本地 Toolkit，核对 `nvcc` 与 PyTorch 的 CUDA 版本，并确认编译产物包含可在目标 GPU 上运行的代码。二进制与 PTX 的兼容关系参见 [NVIDIA Blackwell 兼容指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)。本项目暂不锁定未经实测的依赖组合。

在目标机器已有合适的 Python / PyTorch / CUDA 环境后，可从以下命令开始：

```bash
# 以下命令均从本项目根目录执行。
nvidia-smi
nvcc --version
python -m torch.utils.collect_env

python -c 'import torch; p = torch.cuda.get_device_properties(0); print(p); print("CC:", torch.cuda.get_device_capability(0)); print("SMs:", p.multi_processor_count); print("PyTorch CUDA:", torch.version.cuda)'

# setup.py 在构建时导入 torch，因此使用已有环境中的构建依赖。
python -m pip install setuptools wheel ninja numpy
python -m pip install --no-build-isolation -v ./third_party/marlin
python third_party/marlin/test.py
```

这些是待目标机器验证的起点，不是已通过验证的安装配方。先保存编译日志和测试结果；如需兼容性修改，单独记录补丁和原因。

### 5.2 原始 benchmark 的已知限制

固定版本的 `bench.py` 只识别部分旧 GPU，遇到 5090 时会设置 `SMS=-1`，随后用它构造 `ideal` shape，产生负数维度。因此 **不能直接把 `python third_party/marlin/bench.py` 当作可用的 5090 基线命令**。

后续基线驱动放在 [`benchmarks/`](benchmarks/README.md)：读取实际设备属性、选择适合显存的 shape，并调用未修改的上游内核。若选择修改原始脚本，应将补丁保存到 `experiments/`，明确只是测量脚本适配。

另外，原始 `test.py` 的 `test_llama_shapes` 提前 `return`；测试通过不代表这些模型形状被验证。原始 `bench.py` 使用合成权重和零 scale 计时，也不能替代正确性测试。

### 5.3 建立可比较的实验口径

1. **正确性先行**：先通过原始有效测试，再为实际计时的 shape 检查同一量化权重下的输出；覆盖 groupsize `-1` 和 `128`。
2. **先小后大**：从 `M ∈ {1, 2, 4, 8, 16, 32, 64, 128}`、`K=N=4096` 开始，再增加模型中的投影 shape。高层 `Layer` 要求 K 可被 128 整除、N 可被 256 整除；低层支持范围还与内核配置有关。
3. **两条参照线**：同时记录原始 Marlin 和 FP16 `torch.matmul` 的耗时。前者用于衡量优化收益，后者用于观察量化算子的相对性能。
4. **计时口径一致**：权重打包、数据生成和首次编译移出内核计时；单独说明是否包含分配和 launch 开销。预热后用 CUDA events 或明确同步的计时方法，多轮测量并报告中位数和波动。
5. **控制 Laptop 状态**：插电并固定性能模式，记录温度、时钟和功耗；区分短时结果与持续运行结果。保持相同的缓存与计时策略，对比时交替测量基线和优化版本。
6. **保存复现信息**：本项目提交、上游提交、补丁、完整命令、环境、shape、groupsize、精度指标、延迟和测量次数一起保存到 [`results/`](results/README.md)。

如果内核本身必须修改才能运行，结果标记为“兼容性适配基线”，同时保留原始版本失败的日志。编译通过、正确性通过、性能改善是三个独立结论。

## 6. 优化与 Desktop 验证

基线稳定后再用 Nsight Compute 等工具定位瓶颈：显存带宽、L2 命中率、共享内存访问、寄存器占用、Tensor Core 利用率、反量化指令和归约开销。依据测量选择优化方向，每次实验只改变一个主要变量。

候选方向包括 tile / warp 配置、流水线深度、访存布局、反量化与计算的调度、任务划分与归约。它们是待验证的假设，不预设对 5090 必然有效。每项实验必须同时报告正确性、收益和退化 shape。

Desktop 阶段重新采集设备信息、运行原始或同一适配基线，再验证 Laptop 上的优化。允许针对设备重新选择参数，不直接沿用 Laptop 的性能结论。

## 7. 里程碑

- [x] 创建项目结构，引入并固定官方 Marlin 源码。
- [ ] 完成接口、权重布局、Tensor Core、流水线和归约笔记。
- [ ] 在 5090 Laptop 上验证环境、编译和正确性。
- [ ] 完成 5090 benchmark 适配，保存首份可复现基线。
- [ ] 完成瓶颈分析与至少一项经过验证的优化实验。
- [ ] 在 5090 Desktop 上复测并总结可迁移性。

## 8. 参考与许可

- [原始 Marlin 仓库](https://github.com/IST-DASLab/marlin)及其[使用说明](third_party/marlin/README.md)。
- [MARLIN 论文：Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743)。
- 上游代码采用 [Apache-2.0](third_party/marlin/LICENSE)，保留其许可证和版权声明；后续派生实现也需保留相应声明。
