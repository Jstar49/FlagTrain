"""FlagTrain: a compatible reimplementation of DeepSpeed.

The operator layer lives in ``flag_train.deepspeed`` and mirrors the public names
of ``deepspeed.inference.v2.kernels.ragged_ops``, so a caller can swap the import
-- ``from flag_train.deepspeed import blocked_flash`` -- rather than rewrite the
call. The promised names are ``flag_train.deepspeed``'s ``__all__``; the upstream
versions and hardware actually verified are in ``docs/compatibility.md``.
"""

# ``deepspeed`` and ``megatron`` are imported for the attributes after a bare
# ``import flag_train``; ``testing`` is a re-export and ``runtime`` is used just
# below.
from flag_train import (  # noqa: F401
    deepspeed,
    megatron,
    runtime,
    testing,
    transformer_engine,
)

# Vendor-operator registration: write each backend's specialised operators over
# the same-named generic ones, once every package above has bound its names.
# This is FlagGems' layout, and it is load-bearing. Scanning the backend's
# operator packages imports them, and an override imports the generic package it
# replaces, so registering from inside a subpackage (``flag_train.megatron`` used
# to) re-entered the scan while that package was still half imported: the scan
# saw no ``__all__``, cached nothing for it, and the package kept its generic
# kernels while ``get_customized_ops`` later reported the override as available.
# Registering here means the scan only ever meets complete packages.
from flag_train.runtime.backend import SpecOpRegistrar

device = runtime.device.name
vendor_name = runtime.device.vendor_name

for _ops_package in (deepspeed, megatron, transformer_engine):
    SpecOpRegistrar(registry=_ops_package.__dict__, vendor=vendor_name).apply()
del _ops_package


class use_train:
    """Placeholder for FlagTrain's aten-patching context manager.

    The real ``use_train`` patches ``torch.ops.aten`` for the duration of the
    block, which needs an aten-patch registry (``torch.library.Library``) that
    FlagTrain does not carry yet. That is a different thing from the vendor
    operator registry, which runs at the end of this module's import (see the
    ``SpecOpRegistrar`` loop above). Nothing in this repository enters this class
    -- the benchmarks pass an explicit ``gems_op`` -- so it only has to exist and
    fail loudly if a caller assumes the FlagTrain behaviour.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flag_train has no registered aten operator overrides yet; pass the "
            "operator under test explicitly instead of relying on use_train()."
        )


# The operators are deliberately absent: they are reached through
# ``flag_train.deepspeed``, which is the one place their names are enumerated and
# the place the vendor registration rewrites. This exports only what the top level
# itself defines.
__all__ = [
    "device",
    "use_train",
    "vendor_name",
]
