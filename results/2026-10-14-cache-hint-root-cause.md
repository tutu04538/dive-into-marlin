# Marlin 非法指令根因定位（2026-10-14）

**本机根因定位到 CUDA 13.0.88 的 `ptxas` 优化代码生成：Marlin 的 `cp_async4_stream` 使用 cache policy 的异步搬运路径，被编译为引用未初始化 uniform register 的异常 SASS。** 移除 cache hint 的隔离副本通过了全部 35 个有效上游计算用例；保持原始内核不变、仅关闭 `ptxas` 优化，纯 CUDA 复现也通过。

这是本机实验和反汇编支持的诊断；尚未获得 NVIDIA 对具体编译器内部缺陷的确认。本次没有性能测量，也没有修改 `third_party/marlin` 或覆盖 `.venv` 中安装的原始扩展。

## 环境

源码为上游 `1f25790bdd49fba53106164a24666dade68d7c90`，项目诊断前提交为 `7f674f008422976a146cd5696fbfb65cbdf2fdea`。环境沿用[首次测试记录](2026-10-12-5090-laptop-local-test.md)：RTX 5090 Laptop、CC 12.0、WSL2、驱动 592.01、Python 3.12.14、PyTorch 2.13.0+cu130。`nvcc` 和 `ptxas` 均为 13.0.88；`cuobjdump` 为 13.0.85。目标始终为 `sm_120`。

## 对入门开发者的解释

原始 Marlin 从显存搬运权重和 scale 到 shared memory 时，除了指定“搬运 16 字节”，还附加了“这些数据可以优先从 L2 缓存淘汰”的提示：

```text
createpolicy ... evict_first  → 生成 64-bit 缓存策略
cp.async ... cache_hint      → 异步搬运时使用该策略
```

CUDA 编译分为 C++ → PTX → SASS（GPU 机器指令）几个阶段。本机在 PTX 中能看到正确的策略创建和使用，但优化后的 SASS 有一部分搬运指令引用了没有定义的寄存器，导致执行时出现非法指令错误。调用点之后的 `torch.cuda.synchronize()` 才把 GPU 错误报告给 Python。

根据 [NVIDIA PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)，cache policy 是性能提示，不改变程序的内存一致性语义。因此移除提示、保留异步搬运本身是一个合理的单变量对照；它的性能影响仍需单独测量。

## 证据链

### 1. 原始错误能够在纯 CUDA 中复现

新增[纯 CUDA 复现程序](../experiments/001-cache-hint-codegen/standalone.cu)，直接包含未修改的上游 `.cu` 文件，完全不加载 PyTorch。固定 `(M,N,K)=(16,256,512)`、groupsize 128、thread tile `(128,128)`，设置全部 A、反量化后的权重和 scale 为 1，期望所有输出为 512。

原始构建结果：

```text
marlin_cuda return: 0
cudaDeviceSynchronize(): an illegal instruction was encountered (715)
```

该复现不使用 Python 包装层或其 workspace dtype，也不依赖随机输入。此次错误可以在内核的独立 CUDA 执行路径中触发。

### 2. PTX 正常保留了缓存策略指令

生成原始 PTX：

```bash
nvcc -arch=compute_120 -ptx -std=c++20 --expt-relaxed-constexpr \
  third_party/marlin/marlin/marlin_cuda_kernel.cu \
  -o results/raw/2026-10-14-root-cause/original.ptx
```

在首次失败对应的 `Marlin<256,1,8,8,4,8>` 实例中，统计到 30 条 `createpolicy.fractional` 和 30 条 `cp.async.cg.shared.global.L2::cache_hint`。策略在 PTX 中不是未定义的输入。

### 3. 优化后机器码出现了没有初始化的寄存器引用

对原先安装的 `marlin_cuda` 扩展和纯 CUDA 原始复现程序分别使用 `cuobjdump --dump-sass`。两者对应实例均含如下指令：

```text
/*1130*/ LDGSTS.E.BYPASS.LTC128B.128 [R7+UR0+0x4000], desc[UR1][R92.64];
/*1140*/ LDGSTS.E.BYPASS.LTC128B.128 [R7+UR0+0x5000], desc[UR1][R90.64];
```

该实例有 30 处 `desc[UR1]` 使用，未找到对 `UR0` / `UR1` 的写入。相邻代码仍可见在其他寄存器中构造描述符的指令，因此这里的准确描述是“生成的描述符/地址寄存器引用异常”，不能简单说所有 `createpolicy` 都被删除。

此外，原始扩展中的 `Marlin<256,1,16,4,4,-1>` 实例也有 21 处 `desc[UR1]` 使用且无对应寄存器写入，与前次非分组量化路径同样失败的现象一致。没有声称所有内核实例都出现这一问题。

### 4. 只关闭 ptxas 优化，原始内核能够正确运行

同一纯 CUDA 程序、相同输入与编译参数，只增加 `-Xptxas=-O0`：

```text
marlin_cuda return: 0
standalone Marlin: PASS (all outputs equal 512)
```

cache hint 和原始内核代码均保留。此版本对应实例的 SASS 没有 `desc[UR1]` 异常使用。这把故障进一步缩小到 `ptxas` 的优化代码生成路径，而非 PTX 指令在该设备上一概不可执行。

这里仅验证了纯 CUDA 单个复现 shape，不把 `-O0` 构建宣称为已通过完整测试的推荐版本，也没有测量其性能。

### 5. 只移除 cache hint，上游正确性测试通过

在导出的隔离副本中，仅对 [`cp_async4_stream`](../third_party/marlin/marlin/marlin_cuda_kernel.cu) 使用如下变化：

```diff
-    "   .reg .b64 p;\n"
-    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
-    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
+    "   cp.async.cg.shared.global [%0], [%1], %2;\n"
```

这同时影响该 helper 搬运的权重和 scale；不改 MMA、反量化、tile、流水线等待或归约。使用相同 PyTorch 环境和 `sm_120` 目标构建，通过命令级 `PYTHONPATH` 加载对照扩展：

```text
Ran 6 tests
OK
```

实际执行 35 个正常计算用例，涵盖 groupsize `-1` / `128`、两个 thread tile 配置和上游 `test_tiles` 的 M 范围。原始误差断言均通过；`test_errors` 也通过。`test_llama_shapes` 仍是提前返回，没有测试对应模型 shape。

编译时的除零 / 取模警告在此版本仍出现。因此不能把这些警告直接当成本次非法指令的根因。

随后去掉 `PYTHONPATH` 再测原始扩展的 `test_groups`，仍报相同非法指令错误，确认原始失败和对照成功是不同构建的实际结果。

### 6. 简单 cache-hint 搬运本身可以运行

[搬运对照程序](../experiments/001-cache-hint-codegen/copy_probe.cu)分别执行不带 hint 和带同一 `evict_first` hint 的 16 字节异步搬运，两种模式均正确复制全部数据。这支持“复杂内核上下文触发的代码生成问题”，不支持“Blackwell 完全不支持 cache hint”的判断。

## 对照汇总

| 构建 / 实验 | 源码变化 | 实测结果 |
| --- | --- | --- |
| 原始扩展，默认 ptxas 优化 | 无 | `test_groups` 非法指令 |
| 纯 CUDA 原始 Marlin，默认 ptxas 优化 | 无内核修改 | 错误 715 |
| 纯 CUDA 原始 Marlin，`-Xptxas=-O0` | 无内核修改 | 全部输出为 512 |
| 隔离扩展，默认 ptxas 优化 | 只移除 cache hint | 35 个有效计算用例通过 |
| 简单搬运对照 | 分别带 / 不带 hint | 两种模式均通过 |

## 外部证据与限制

- 上游 [issue #40](https://github.com/IST-DASLab/marlin/issues/40) 报告了 RTX 5090 / CUDA 12.8 的同类症状，作者后来采用移除 cache policy 的绕过方式。本次独立在本机验证了这一变化。
- 上游 [issue #44](https://github.com/IST-DASLab/marlin/issues/44) 报告了 CUDA 13.1 / `sm_90` 的异常 `LDGSTS` 描述符引用。它与本机观察相似，但环境不同，不将其作为本机根因的唯一依据，也不直接套用其“策略被删除”的解释。
- 本机 Compute Sanitizer 报告 WDDM 调试接口未启用，无法获得有效的设备级诊断。因此没有声称拿到了精确的 GPU trap PC，或已经排除了全部潜在内存和同步问题。结论依据是反汇编、纯 CUDA 复现和两种单变量对照。
- 尚未定位 `ptxas` 内部具体出错的优化 pass，没有测试其他 Toolkit / 驱动版本或 Desktop。兼容性绕过只在上述范围得到验证。

## 产物与后续使用

复现程序、最小补丁及构建命令见 [`experiments/001-cache-hint-codegen`](../experiments/001-cache-hint-codegen/README.md)。这次保留原始源码和安装不变，兼容性版本仅由命令级 `PYTHONPATH` 选用。

原始日志、PTX、SASS、隔离源码和二进制保存在本机 `results/raw/2026-10-14-root-cause/`，被 Git 忽略。其中包括 `memcheck-original.log`、`no-hint-build.log`、`no-hint-tests.log`、`original-recheck.log`、`standalone-original.log`、`standalone-O0.log`、`original.ptx`、`grouped-original.sass`、`standalone-original.sass` 和 `standalone-O0.sass`。报告与复现源码可随项目保存。

若后续建立基线，需先明确选择“原始构建”还是“移除 cache hint 的兼容性适配版本”。移除提示可能改变缓存行为，不能把适配版本的性能称为原始版本的性能。
