# 001：定位 cache hint 路径的代码生成错误

类别：兼容性诊断。基于上游 `1f25790bdd49fba53106164a24666dade68d7c90`，目标为本机 RTX 5090 Laptop / `sm_120` / CUDA 13.0.88。

已验证：原始 Marlin 的优化后机器码在 `cp_async4_stream` 对应路径出现异常的寄存器引用；关闭 `ptxas` 优化或在隔离副本中移除 cache hint 均能绕过。完整证据和结论边界见[根因记录](../../results/2026-10-14-cache-hint-root-cause.md)。这不是性能优化实验，没有性能数据。

## 文件

- `standalone.cu`：包含原始 Marlin 内核的纯 CUDA 复现程序。A 全 1、权重 nibble 全 9（反量化为 1）、scale 全 1，检查 `(M,N,K)=(16,256,512)` 的全部输出等于 512。只执行一次矩阵乘法。
- `copy_probe.cu`：单条 cache-hint 搬运路径的对照程序，分别检查带 hint / 不带 hint 的 256×16 字节搬运结果。两者都通过，说明简单场景中该指令路径可运行。
- `remove-cache-hint.patch`：只在 `cp_async4_stream` 中移除 policy 创建及 `.L2::cache_hint`，保持 16 字节 `cp.async.cg.shared.global` 搬运及其他计算逻辑不变。保留上游许可证和版权声明。

## 复现纯 CUDA 对照

以下从项目根目录执行，需要当前 `nvcc` 支持 `sm_120`。编译器警告可能出现，原始编译日志已保存在本机 raw 目录。

```bash
mkdir -p results/raw/cache-hint-repro

nvcc -arch=sm_120 -lineinfo -std=c++20 --expt-relaxed-constexpr \
  -Ithird_party/marlin/marlin experiments/001-cache-hint-codegen/standalone.cu \
  -o results/raw/cache-hint-repro/original

nvcc -arch=sm_120 -lineinfo -std=c++20 --expt-relaxed-constexpr -Xptxas=-O0 \
  -Ithird_party/marlin/marlin experiments/001-cache-hint-codegen/standalone.cu \
  -o results/raw/cache-hint-repro/ptxas-O0

# 本机原始版本预期退出 1，CUDA 同步报错 715；单独运行下一条命令。
results/raw/cache-hint-repro/original
# 本机预期退出 0，全部输出为 512。
results/raw/cache-hint-repro/ptxas-O0

nvcc -arch=sm_120 -lineinfo -std=c++17 \
  experiments/001-cache-hint-codegen/copy_probe.cu \
  -o results/raw/cache-hint-repro/copy-probe
results/raw/cache-hint-repro/copy-probe 0
results/raw/cache-hint-repro/copy-probe 1
```

`-Xptxas=-O0` 仅用于定位代码生成问题，没有测量其性能，也没有将其设为项目默认构建方式。

## 复现最小绕过补丁与原始测试

以下使用已有的项目 `.venv`（Python / PyTorch 版本见测试记录），把构建和补丁留在独立目录。首次执行时使用空的 `source` 目录。

```bash
mkdir -p results/raw/cache-hint-repro/source
git -C third_party/marlin archive 1f25790bdd49fba53106164a24666dade68d7c90 \
  | tar -x -C results/raw/cache-hint-repro/source
patch -d results/raw/cache-hint-repro/source -p1 \
  < experiments/001-cache-hint-codegen/remove-cache-hint.patch

(
  cd results/raw/cache-hint-repro/source
  TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
    ../../../../.venv/bin/python setup.py build_ext \
    --build-lib ../lib --build-temp ../build
)

PYTHONPATH="$PWD/results/raw/cache-hint-repro/lib" \
  .venv/bin/python -c 'import torch, marlin_cuda; print(marlin_cuda.__file__)'
PYTHONPATH="$PWD/results/raw/cache-hint-repro/lib" \
  .venv/bin/python -u third_party/marlin/test.py -v
```

本机结果为 6 个测试方法 `OK`，其中实际执行 35 个正常计算用例；`test_llama_shapes` 仍提前返回，未覆盖其模型形状。测试使用上游误差断言 `mean(abs(C-C_ref)) / mean(abs(C_ref)) < 0.001`。

此方式不重新安装 `.venv` 中的原始 Marlin，也不修改 submodule。去掉命令前的 `PYTHONPATH=...` 即恢复加载原始扩展。实际诊断文件位于 `results/raw/2026-10-14-root-cause/`，上述命令使用另一目录以免覆盖这些证据。

cache hint 的改变可能影响性能，当前只验证了兼容性与所覆盖用例的正确性。后续若采用，应称为兼容性适配版本，并保留原始版本的失败记录。
