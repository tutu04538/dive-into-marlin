# 5090 Laptop 本地可运行性测试（2026-10-12）

**结论：未修改的原始 Marlin 在本机编译、安装、导入成功，但正常计算触发 CUDA 非法指令错误，正确性验证未通过。** 本次仅测试可运行性，没有运行 `bench.py`、采集内核性能或建立性能基线，没有修改内核或应用兼容性补丁。

## 环境与源码

| 项目 | 实测值 |
| --- | --- |
| 项目测试前提交 | `8dc271d31f6ba2957f125211f9b7981e79ef1dee` |
| 上游 Marlin 提交 | `1f25790bdd49fba53106164a24666dade68d7c90`，源码未修改 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| Compute capability / SM 数 | 12.0 / 82 |
| PyTorch 报告的显存 | 25,650,855,936 bytes（约 23.89 GiB） |
| 操作系统 | Ubuntu 22.04.5 LTS，WSL2 |
| 内核 | `6.6.87.2-microsoft-standard-WSL2` |
| NVIDIA 驱动 | 592.01（`nvidia-smi` 显示的 CUDA 13.1 是驱动支持信息） |
| CUDA Toolkit / nvcc | `/usr/local/cuda`，13.0.88 |
| Python | 3.12.14 |
| PyTorch | `2.13.0+cu130`，`torch.version.cuda == '13.0'` |
| NumPy | 2.3.5 |
| GCC | 11.4.0 |
| 构建依赖 | setuptools 80.10.2，wheel 0.47.0，ninja 1.13.2 |
| 测试虚拟环境中的 pip | 25.0.1 |
| 编译目标 | `TORCH_CUDA_ARCH_LIST=12.0`，日志确认 `-gencode=arch=compute_120,code=sm_120` |
| 编译并行度 | `MAX_JOBS=4` |

基础环境为本机现有的 Conda `vllm-omni-dev`，项目 `.venv` 通过 `--system-site-packages` 复用其依赖。因此 `.venv` 并非完全独立的依赖环境；复现需准备上表中的实际版本。本次没有安装或升级基础环境依赖，Marlin 仅安装到项目 `.venv`。

## 执行过程与复现

以下命令从项目根目录执行。先通过 `nvidia-smi`、`nvcc --version`、`python -m torch.utils.collect_env` 采集环境，并用 `torch.cuda.get_device_properties(0)` 检查设备。一个随机 32×32 FP16 `torch.matmul` 经有限值检查和 CUDA 同步后成功。

为避免构建产物污染 submodule，从固定提交导出未经修改的源码副本：

```bash
mkdir -p results/raw/2026-10-12-local-test/source
git -C third_party/marlin archive HEAD | tar -x -C results/raw/2026-10-12-local-test/source
python -m venv --system-site-packages .venv

.venv/bin/python -m torch.utils.collect_env \
  > results/raw/2026-10-12-local-test/environment.log 2>&1

TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
  .venv/bin/python -m pip install --no-deps --no-build-isolation -v \
  ./results/raw/2026-10-12-local-test/source \
  > results/raw/2026-10-12-local-test/build.log 2>&1

.venv/bin/python -u third_party/marlin/test.py -v \
  > results/raw/2026-10-12-local-test/test.log 2>&1
```

构建命令退出码为 0，成功安装 `marlin-0.1.1`。编译器对上游模板代码报告了 `%` 右操作数为零、除零和未使用变量警告，尚未确认其与运行错误存在因果关系。

在项目根目录单独导入，确认 `marlin` 和 `marlin_cuda` 均来自 `.venv/lib/python3.12/site-packages/`；原始测试脚本所在目录也含上游 `marlin` Python 包，两者源码一致，CUDA 扩展来自本次安装。

## 原始测试与独立复测结果

完整测试退出码为 1，摘要为：

```text
Ran 6 tests
FAILED (errors=4)
```

`test_groups` 首次正常内核调用后在 `test.py:66` 的 `torch.cuda.synchronize()` 抛出：

```text
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

完整测试中，后续测试在创建 CUDA 张量时继续报错，可能受同进程中先前错误影响。因此又将 4 个计算测试方法分别放到全新的 Python 进程，并设置 `CUDA_LAUNCH_BLOCKING=1`。复测方式如下，每个方法单独启动进程：

```bash
for test in test_groups test_tiles test_k_stages_divisibility test_very_few_stages; do
  CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -u third_party/marlin/test.py "Test.$test" -v \
    > "results/raw/2026-10-12-local-test/$test.log" 2>&1
done
```

| 独立测试方法 | 首个失败的 `(M, N, K)` | `(thread_k, thread_n)` | groupsize | 结果 |
| --- | --- | --- | --- | --- |
| `test_groups` | `(16, 256, 512)` | `(128, 128)` | 128 | 非法指令，退出码 1 |
| `test_tiles` | `(1, 512, 1024)` | `(64, 256)` | -1 | 非法指令，退出码 1 |
| `test_k_stages_divisibility` | `(16, 512, 768)` | `(64, 256)` | -1 | 非法指令，退出码 1 |
| `test_very_few_stages` | `(16, 512, 64)` | `(64, 256)` | -1 | 非法指令，退出码 1 |

独立复测均在 `marlin.mul(...)` 后的 CUDA 同步处报错，未进入输出误差断言。各方法在首个失败用例就终止，不能声称其后续 shape 已被测试。

另外两个测试方法的含义：

- `test_errors`：完整测试中显示 `ok`，仅表示异常断言满足，不代表正常矩阵乘法成功。
- `test_llama_shapes`：上游函数开头提前 `return`，虽显示 `ok`，实际没有执行模型形状测试。

## 结论边界与日志

该结果只说明此固定源码、工具链及 `sm_120` 原生构建在本机未通过测试。根因尚未定位，没有测试其他 CUDA / PyTorch 组合或其他编译目标，也没有尝试代码修复。无法据此判断 Marlin 在所有 5090 环境中的兼容性。

原始日志保存在本机 `results/raw/2026-10-12-local-test/`：`environment.log`、`build.log`、`modules.log`、`test.log` 以及四份 `test_*.log`。该目录被 Git 忽略；远端保留本记录中的环境、命令、错误摘要和失败用例，不含构建产物或本地虚拟环境。
