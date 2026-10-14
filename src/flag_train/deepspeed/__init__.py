"""DeepSpeed-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .blocked_flash import blocked_flash
from .lamb import lamb
from .packbits import packbits

__all__ = [
    "blocked_flash",
    "lamb",
    "packbits",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
