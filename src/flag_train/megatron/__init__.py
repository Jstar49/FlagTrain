"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .fused_proj_rms_compute_h import fused_proj_rms_compute_h_fwd

__all__ = [
    "fused_proj_rms_compute_h_fwd",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
