"""Megatron-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

__all__ = []


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
