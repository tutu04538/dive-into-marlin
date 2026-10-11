# 第三方源码

`marlin/` 是 Git submodule，来源为官方仓库 <https://github.com/IST-DASLab/marlin>。

| 项目 | 当前记录 |
| --- | --- |
| 固定提交 | `1f25790bdd49fba53106164a24666dade68d7c90` |
| 上游提交日期 | 2024-09-04 |
| 上游提交说明 | Update README.md to point to arxiv paper |
| 许可证 | [Apache-2.0](marlin/LICENSE) |

submodule 内保持原始源码，作为阅读和性能对照的锚点。兼容性补丁和优化实验放到父仓库的 `experiments/` 中。

在项目根目录执行 `git submodule update --init --recursive` 恢复父仓库指定的版本。不要使用 `git submodule update --remote` 无意更新基线。确实需要升级上游时，单独提交版本变化，更新这里和根 README 的记录，并重新建立基线。
