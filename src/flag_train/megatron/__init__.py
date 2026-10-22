"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .quick_gelu import quick_gelu

__all__ = [
    "quick_gelu",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
