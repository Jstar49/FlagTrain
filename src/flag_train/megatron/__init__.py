"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .fused_gated_norm import fused_gated_norm
from .fused_indices_to_multihot import fused_indices_to_multihot
from .fused_pad_routing_map import fused_pad_routing_map
from .quick_gelu import quick_gelu

__all__ = [
    "fused_gated_norm",
    "fused_indices_to_multihot",
    "fused_pad_routing_map",
    "quick_gelu",
]

SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
