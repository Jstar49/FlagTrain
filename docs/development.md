# 开发说明

当前仓库是初始骨架，三个组件包只有说明性 docstring。没有算子、训练流程、测试或
性能基准的实现，也没有已验证的上游版本或硬件组合。

## 文件归属

按组件 `megatron`、`deepspeed`、`transformer_engine` 放置新增内容：

| 内容 | 位置 |
| --- | --- |
| 实现 | `src/flag_train/<组件>/` |
| 正确性测试 | `tests/<组件>/` |
| 性能基准 | `benchmark/<组件>/` |
| 使用示例 | `examples/<组件>/` |
| 使用方式、开发约定、验证记录 | `docs/` |
| 开发辅助脚本 | `tools/` |

每次贡献应围绕具体实现，同时补充必要的测试、文档和可复现步骤。当前包入口不导入
上游框架；新增功能时再明确真实依赖和加载方式，避免仅导入包就要求安装所有组件。
跨组件的代码组织调整先在 Issue 中说明范围与理由。

## 正确性验证

为新增算子或执行路径指定参考实现、版本和输入条件，比较前向输出及输入、参数梯度。
说明数据类型、误差容限、随机种子和边界输入；低精度路径应验证对应的精度及数值稳定性。
涉及分布式执行时，记录硬件、进程数、并行或分片配置以及通信条件。

测试应能发现实际功能偏差。文档目录或导入检查不能替代算子正确性测试，也不要加入
总是通过的占位测试。首个真实测试应同时给出依赖、运行命令和实际结果。

## 性能验证

在正确性验证通过后再报告性能。参考实现与新实现在相同硬件、软件栈、输入形状、批量、
数据类型、精度策略和并行配置下比较；记录预热、重复次数、同步方式、计时范围、统计
方法及峰值显存。区分编译或初始化开销、算子耗时和端到端吞吐，不将局部结果外推为
训练性能，不把不同精度或不同有效批量的结果直接比较。

提交时区分已完成与未完成的验证，把可复现证据和限制补充到
[兼容性记录](compatibility.md)。协作流程见 [贡献指南](../CONTRIBUTING.md)。

## 运行测试

按算子选择（每个算子一个 pytest marker，与 FlagGems 的用法一致）：

```sh
# 正确性：参考实现跑在 CPU 上（--ref cpu），结果写入指定文件
pytest -m "lamb" --record json --output accuracy_lamb.json \
    --ref cpu --continue-on-collection-errors -vs

# 性能：--level core 用各算子的核心 shape（注意要给出 benchmark 路径）
pytest benchmark -m "lamb" --level core --record json \
    --output benchmark_lamb.json --continue-on-collection-errors
```

* 算子名就是 marker 名（当前：`lamb`、`blocked_flash`、`packbits`）。用
  `pytest -m "lamb or blocked_flash" --collect-marks marks.yaml --collect-only`
  可以把全部 marker 及其测试文件导出来；也可用 `pytest -m "<算子>" --collect-only -q`
  先看选中哪些用例。
* **benchmark 必须带 `benchmark` 路径**（`pytest benchmark ...`）。`--level` 等选项由
  `benchmark/conftest.py` 声明，而 pytest 在**收集之前**就解析命令行：没有路径参数时
  只收集 `testpaths`（即 `tests/`），benchmark 的 conftest 根本不会被加载，于是报
  `unrecognized arguments: --level`。**FlagGems 完全一样** —— 实测在其仓库里跑
  `pytest -m "linalg_cross" --level core`，报错与 FlagTrain 逐字相同；带上 `benchmark`
  就正常（`1/2829 collected`）。它们文档里看着"不用带路径"的那条命令是跨行书写的，
  路径在上一行。功能测试则确实不用带路径（`testpaths` 正好就是 `tests/`）。
* 两个套件的选项分工：`--ref cpu` 属于 `tests/`（功能套件，让参考实现在 CPU 上真正执行，
  与设备上的算子实现独立）；benchmark 侧用 `--mode`（`kernel`/`operator`/`wrapper`/
  `cudagraph`）选择被测实现。`--record json --output <文件>` 两边都支持，默认文件名分别是
  `accuracy_result.json` 与 `benchmark_result.json`；`--record log` 则写 `result_*.log`。
* `--continue-on-collection-errors` 在某个模块导入失败时（例如某平台缺少参考实现所需的包）
  仍继续收集其余用例，`-vs` 打印每个用例名与输出，适合定位单平台问题。
