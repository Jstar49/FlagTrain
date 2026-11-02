"""DeepSpeed-related implementations."""

from .blocked_flash import blocked_flash
from .evoformer_attn import evoformer_attn
from .lamb import lamb
from .lion import multi_tensor_lion
from .mixed_moe_gemm import mixed_moe_gemm
from .multi_tensor_adam import multi_tensor_adam
from .packbits import packbits
from .unpackbits import unpackbits
from .wf6af16_linear import wf6af16_linear

__all__ = [
    "blocked_flash",
    "evoformer_attn",
    "lamb",
    "mixed_moe_gemm",
    "multi_tensor_adam",
    "multi_tensor_lion",
    "packbits",
    "unpackbits",
    "wf6af16_linear",
]

# The vendor-operator registration for these names runs in flag_train/__init__.py
# once this package and its siblings are fully imported -- see the comment there.
