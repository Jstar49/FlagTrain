# FlagTrain

[English](README.md) | 中文

## 介绍

FlagTrain 是 [FlagOS](https://flagos.io/) 的一部分。FlagOS 是面向多元 AI 芯片的开源系统软件栈，致力于打通模型、系统与芯片之间的软件协作，降低模型在不同硬件平台上的迁移与维护成本。

FlagTrain 是一个基于 [Triton](https://github.com/triton-lang/triton) 语言实现的训练算子与计算组件库，面向主流训练框架及关键组件，提供可复用、可优化的训练计算实现，旨在加速大模型在多种硬件后端上的训练。

对于训练框架开发者，FlagTrain 致力于提供符合相应接口和计算语义的 Triton 实现，减少重复开发与硬件适配工作；对于算子开发者，FlagTrain 提供围绕实际训练场景组织实现、测试和性能优化的协作平台。

项目处于持续建设阶段，文档与组件实现会逐步补充；组件及硬件兼容性尚未完成全面验证。

## 安装

FlagTrain 不发布也不维护自己的容器镜像。公共 FlagOS 软件栈由
[flagos-ai/build-infra](https://github.com/flagos-ai/build-infra) 根据其
[`configs.yaml`](https://github.com/flagos-ai/build-infra/blob/main/configs.yaml)
统一构建。推荐直接使用公共的 Megatron app 镜像（`megatron_training`），它由
FlagOS runtime 镜像加上 `megatron-core` 构成：

| 后端 | 镜像 |
| --- | --- |
| `nvidia-cuda133` | `harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-generic-13.3:2.2.0-0.2.3` |

镜像名和 tag 由 build-infra 维护。上表对应 `2.2.0-0.2.3` 发布，最新列表见
<https://flagos-ai.github.io/release-info/>。

在镜像内只需安装 FlagTrain 自身依赖：

```bash
python -m pip install -e .           # 使用
python -m pip install -e ".[test]"   # 开发与测试
```

FlagOS 网络内安装前需把 `PIP_INDEX_URL` 指向镜像源（镜像内的 pip 没有持久化该
配置），详见[安装说明](docs/installation.md)。

厂商分组按 `backends.yaml` 安装对应厂商的软件栈，适用于没有使用 FlagOS 镜像的
环境：

```bash
python -m pip install -e ".[metax]"        # 沐曦
python -m pip install -e ".[test,nvidia]"  # NVIDIA + 测试依赖
```

镜像已经提供 torch、numpy、flagtree 和 megatron-core，不需要再安装厂商分组；
两者版本不一致时以镜像为准。镜像与分组列表、包索引及依赖同步方式见
[安装说明](docs/installation.md)。

## 设计目标

- **训练计算的 Triton 实现**：围绕训练过程中的前向计算、反向传播及相关计算模块，开发可复用的算子实现。
- **融合与性能优化**：结合训练场景，优化算子融合、访存和并行计算，减少中间数据读写与执行开销。
- **多硬件后端适配**：依托兼容的 Triton 编译与运行环境，推进不同硬件后端上的功能适配和性能调优。
- **与上游协同**：对齐所面向框架及组件的接口、计算语义和使用需求，持续跟进上游变化。
- **可验证的实现**：为算子和计算组件配套正确性测试与性能基准，明确适用条件与验证范围。

## 组件范围

| 框架或组件 | 重点关注方向 |
|---|---|
| [Megatron](https://github.com/NVIDIA/Megatron-LM) | 大模型训练中的关键计算模块及相关算子 |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | 训练优化相关算子与融合计算 |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Transformer 计算模块、混合精度与低精度训练相关算子 |

上述项目构成首批建设范围。具体支持情况将按算子、数据类型、组件版本和硬件后端记录在[兼容性说明](docs/compatibility.md)中。

## 与相关项目的关系

FlagTrain 与 FlagOS 生态中的算子库、编译器及训练框架协同建设，各自承担不同职责：

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**：提供通用算子实现。FlagTrain 中可复用的基础算子能力优先与 FlagGems 协同，减少重复实现。
- **FlagTrain**：聚焦训练场景中的算子、融合计算及计算组件，组织相应的实现、测试和性能优化。
- **训练框架与上游组件**：提供应用场景、接口和计算语义参考。FlagTrain 围绕其中的关键计算路径开展 Triton 实现与验证。

组件之间的具体调用和集成方式，将在对应模块文档与使用示例中说明。

## 目录结构

```text
FlagTrain/
├── src/flag_train/
│   ├── megatron/               # Megatron 相关计算实现
│   ├── deepspeed/              # DeepSpeed 相关计算实现
│   └── transformer_engine/     # Transformer Engine 相关计算实现
├── tests/                      # 按组件组织的正确性与接口兼容性测试
├── benchmark/                  # 按组件组织的性能基准
├── examples/                   # 按组件组织的调用与接入示例
├── docs/                       # 安装、开发与兼容性说明
├── tools/                      # 开发与维护工具
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── pyproject.toml
```

三个组件目录用于直接存放后续的 Triton 计算实现与接口封装。组件内部将按具体算子或计算模块组织文件，前向、反向及调用封装可以放在一起。公共能力将在出现明确复用需求时提取。

## 开发与验证

每项算子或计算组件的实现，应提供相应的参考实现、测试用例和性能验证方法。

### 正确性测试

- 覆盖代表性的输入形状、数据类型和边界条件。
- 对照参考实现验证计算结果，明确数值误差要求。
- 对涉及反向传播的实现，验证梯度计算。
- 记录测试使用的组件版本、软件环境和硬件后端。

### 性能测试

- 在一致的硬件、输入和精度条件下进行对比。
- 说明参考实现、预热方式及计时方法。
- 根据计算特点报告执行时间、吞吐量或显存占用。
- 区分单算子性能与端到端训练收益。

本地安装骨架请参考[开始使用](docs/getting_started.md)，开发约定请参考[开发说明](docs/development.md)。算子调用示例和测试命令将随首批实现一同提供。

## 参与贡献

欢迎参与训练算子开发、组件适配、性能优化、测试和文档建设。

- 贡献流程请参考[贡献指南](CONTRIBUTING.md)。
- 问题报告和功能建议请提交至 [GitHub Issues](https://github.com/flagos-ai/FlagTrain/issues)。
- 安全问题请按照[安全报告说明](SECURITY.md)私下反馈。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。涉及第三方代码的部分应保留其原有许可证和版权声明。
