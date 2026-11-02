# NVIDIA 基线算子（baseline）

本目录保存运行算子 NVIDIA 基线采集所需的全部 op 定义文件，均应用于
[ops-baseline](https://github.com/zhaojs12345/ops-baseline) 仓库（上游路径
`ops/flagtrain/`）。

这些模块采集的是 FlagTrain benchmark 在 NVIDIA 上使用的基线实现，不是 FlagTrain
自己的 Triton 实现。每个模块自包含，只依赖 torch / DeepSpeed / Megatron，**不
import `flag_train` 或 `benchmark`**；shape、dtype 和输入构造复制在模块内。

按基线实现所属的库分组：

| 目录 | 算子 |
|---|---|
| `deepspeed/` | `lamb`、`lion`、`mixed_moe_gemm`、`wf6af16_linear`、`packbits`、`unpackbits`、`blocked_flash`、`evoformer_attn` |
| `megatron/` | `quick_gelu`、`weighted_squared_relu_impl` |

每个文件需要与上游仓库的同名文件逐字节一致。新增或修改基线时以本仓库为准，并将文件
原样同步到对应后端子目录。

## 依赖

DeepSpeed / Megatron 是硬依赖：对应模块在 `native()` 加载失败时直接抛错，不会回退到
torch 组合。运行基线前需要安装：

1. **DeepSpeed** — 拉最新 `main` 分支源码安装：

   ```bash
   git clone https://github.com/deepspeedai/DeepSpeed.git
   cd DeepSpeed
   pip install -e .
   ```

2. **deepspeed-kernels==0.0.1** — DeepSpeed 预编译 kernel 包：

   ```bash
   pip install deepspeed-kernels==0.0.1
   ```

3. **nvidia-cutlass==3.8.0.0** — DeepSpeed CUTLASS 算子（如 `mixed_moe_gemm`）依赖：

   ```bash
   pip install nvidia-cutlass==3.8.0.0
   ```

4. **nvidia-resiliency-ext==0.6.0** — NVIDIA 容错扩展包，Megatron-LM 的异步
   checkpoint（`megatron.core.dist_checkpointing`）和 fault injection 路径依赖它：

   ```bash
   pip install nvidia-resiliency-ext==0.6.0
   ```

5. **Megatron-LM** — 拉 `main` 分支源码安装（提供 `megatron.core`）：

   ```bash
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   pip install -e .
   ```

第 2、3、4 条可以合并成一条命令：

```bash
pip install deepspeed-kernels==0.0.1 nvidia-cutlass==3.8.0.0 \
    nvidia-resiliency-ext==0.6.0
```
