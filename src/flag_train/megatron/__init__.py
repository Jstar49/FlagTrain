"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .fused_indices_to_multihot import fused_indices_to_multihot
from .fused_proj_rms_compute_h import fused_proj_rms_compute_h_fwd
from .quick_gelu import quick_gelu

__all__ = [
    "fused_proj_rms_compute_h_fwd",
    "fused_indices_to_multihot",
    "quick_gelu",
]

SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
