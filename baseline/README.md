# NVIDIA 基线算子（baseline）

本目录保存运行算子 NVIDIA 基线采集所需的全部 op 定义文件，均应用于
[ops-baseline](https://github.com/zhaojs12345/ops-baseline) 仓库（上游路径
`ops/flagtrain/`）。

这些模块采集 NVIDIA 上的上游实现。每个模块自包含，只依赖 Torch、DeepSpeed、
Megatron 或 Transformer Engine，**不 import `flag_train` 或 `benchmark`**。
shape、dtype 和输入构造与对应 benchmark 保持一致。Optest 从本仓库的 `baseline/`
直接读取采集模块。

按基线实现所属的库分组：

| 目录 |
|---|
| `deepspeed/`
| `megatron/`

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

6. **Transformer Engine** — Router 和 L2 normalization backward 需要 TE 的
   PyTorch CUDA 扩展。H800 采集环境需要 CUDA 13 版 PyTorch 和 cuDNN 9.12
   或更新版本。`CUDA_HOME` 应指向带 `nvcc` 和头文件的 CUDA 13 Toolkit。
   采集 L2 backward 时保留 `NVTE_TORCH_COMPILE=1`。

   ```bash
   python3 -m pip install setuptools wheel packaging ninja
   python3 -m pip install --no-build-isolation \
       'transformer-engine[pytorch,core_cu13]==2.21.0'
   export NVTE_TORCH_COMPILE=1
   ```

   安装后检查 Megatron 的 Router GEMM 入口和 TE 的两个 L2 helper。

   ```bash
   python3 - <<'PY'
   from megatron.core.transformer.moe.moe_utils import te_general_gemm
   from transformer_engine.pytorch.jit import (
       l2normalization_backward_fused,
       l2normalization_fwd_fused,
   )

   assert callable(te_general_gemm)
   assert callable(l2normalization_fwd_fused)
   assert callable(l2normalization_backward_fused)
   PY
   ```

第 2、3、4 条可以合并成一条命令：

```bash
pip install deepspeed-kernels==0.0.1 nvidia-cutlass==3.8.0.0 \
    nvidia-resiliency-ext==0.6.0
```

Sinkhorn 采集 Megatron 的编译前向，MTP loss scaler 采集 Megatron 的反向。
mRoPE 使用 Megatron 非交错前向，weighted SwiGLU 使用 Megatron
`WeightedSwiGLUFunction` 默认分支调用的 `weighted_swiglu`。
Router 计入混合输入转为 router dtype 的开销，并使用 Megatron 的 TE GEMM。
L2 采集 TE 的反向，所需的 forward inverse norm 在计时前生成。
