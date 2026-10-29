"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .fast_gelu import fast_gelu
from .fused_gated_norm import fused_gated_norm
from .fused_indices_to_multihot import fused_indices_to_multihot
from .fused_pad_routing_map import fused_pad_routing_map
from .fused_proj_rms_backward import fused_proj_rms_backward
from .fused_proj_rms_compute_h_backward import fused_proj_rms_compute_h_backward
from .fused_proj_rms_compute_h_forward import fused_proj_rms_compute_h_forward
from .fused_proj_rms_forward import fused_proj_rms_forward
from .quick_gelu import quick_gelu

__all__ = [
    "fast_gelu",
    "fused_gated_norm",
    "fused_indices_to_multihot",
    "fused_pad_routing_map",
    "fused_proj_rms_backward",
    "fused_proj_rms_compute_h_backward",
    "fused_proj_rms_compute_h_forward",
    "fused_proj_rms_forward",
    "quick_gelu",
]

SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
