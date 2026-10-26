# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA baseline for blocked_flash.

Self-contained module; depends only on torch and deepspeed. It does not import
``flag_train`` or ``benchmark``. The baseline is fixed to DeepSpeed's
``BlockedFlashAttn``; when DeepSpeed is unavailable it raises instead of
collecting a baseline or falling back to flash-attn / a torch composition.

The input construction (paged cache, atoms, dense KV) and the atom builder are
copied from ``FlagTrain/benchmark/deepspeed/test_blocked_flash.py``; the atom
field offsets come from ``src/flag_train/deepspeed/blocked_flash.py`` (ABI
constants). DeepSpeed is loaded lazily on the first call.
"""

import torch

OP_NAME = "blocked_flash"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True  # the baseline writes the preallocated `out`

# benchmark/deepspeed/test_blocked_flash.py::_BLOCKED_FLASH_SHAPES
_SHAPES = [
    (128, 64, 16, 16),
    (512, 64, 16, 16),
    (1024, 64, 16, 16),
    (2048, 64, 16, 16),
    (4096, 64, 16, 16),
    (128, 128, 16, 16),
    (1024, 128, 16, 16),
    (4096, 128, 16, 16),
    (1024, 128, 32, 8),
    (4096, 128, 32, 8),
    (4096, 128, 64, 8),
]

_SOFTMAX_SCALE = 1.0

# Atom layout, in int32 slots (src/flag_train/deepspeed/blocked_flash.py).
_ATOM_BLOCK_OFFSET = 0
_ATOM_PTR_HIGH = 1
_ATOM_Q_START = 2
_ATOM_Q_LEN = 3
_ATOM_KV_BLOCKS = 4
_ATOM_TOTAL_EXTENT = 5
_ATOM_GLOBAL_Q_IDX = 6
_ATOM_UNUSED = 7
_ATOM_SLOTS = 8

# InflightSeqDescriptor field offsets, in int32 slots; RaggedBatchDescriptor
# holds n_sequences at slot 1.
_SEQ_START_IDX = 0
_SEQ_N_TOKENS = 1
_SEQ_SEEN_TOKENS = 2
_SEQ_STRIDE = 4
_BATCH_N_SEQUENCES = 1


class RaggedBatchWrapper:
    """The slice of DeepSpeed's ``RaggedBatchWrapper`` that ``AtomBuilder`` reads."""

    def __init__(self, seq_params, kv_block_size):
        self._seq_params = list(seq_params)
        self._kv_block_size = kv_block_size
        self.current_sequences = len(self._seq_params)
        self.current_tokens = sum(q_len for q_len, _ in self._seq_params)

        batch_metadata = torch.zeros(
            (2,), dtype=torch.int32, device=torch.device("cpu")
        )
        batch_metadata[int(_BATCH_N_SEQUENCES)] = self.current_sequences
        self._batch_metadata_shadow = batch_metadata

        descriptors = torch.zeros(
            (max(self.current_sequences, 1), int(_SEQ_STRIDE)),
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        block_idx = []
        block_offsets = []
        start_idx = 0
        for i, (q_len, history_len) in enumerate(self._seq_params):
            descriptors[i, int(_SEQ_START_IDX)] = start_idx
            descriptors[i, int(_SEQ_N_TOKENS)] = q_len
            descriptors[i, int(_SEQ_SEEN_TOKENS)] = history_len

            n_blocks = (history_len + q_len + kv_block_size - 1) // kv_block_size
            block_offsets.append(len(block_idx))
            block_idx.extend(range(len(block_idx), len(block_idx) + n_blocks))

            start_idx += q_len

        self._inflight_seq_descriptors_shadow = descriptors
        self._kv_block_offsets = block_offsets
        self._kv_block_idx = torch.tensor(block_idx, dtype=torch.int32)

    def batch_metadata_buffer(self, on_device=False):
        return self._batch_metadata_shadow

    def inflight_seq_descriptors(self, on_device=False):
        return self._inflight_seq_descriptors_shadow[: self.current_sequences]

    def kv_block_offsets(self):
        return self._kv_block_offsets

    def kv_block_idx(self, device=None):
        if device is None:
            return self._kv_block_idx
        return self._kv_block_idx.to(device=device)


class AtomBuilder:
    """Port of DeepSpeed's ``AtomBuilder`` (``ragged_ops/atom_builder``)."""

    def __call__(self, atoms, ragged_batch, q_block_size, kv_block_size):
        if atoms.device != torch.device("cpu"):
            raise RuntimeError("AtomBuilder must be called on tensors")

        batch_meta = ragged_batch.batch_metadata_buffer(on_device=False)
        descriptors = ragged_batch.inflight_seq_descriptors(on_device=False)
        block_offsets = ragged_batch.kv_block_offsets()

        (
            slot_block_offset,
            slot_q_start,
            slot_q_len,
            slot_kv_blocks,
            slot_total_extent,
            slot_global_q_idx,
            slot_ptr_high,
            slot_unused,
        ) = (
            int(_ATOM_BLOCK_OFFSET),
            int(_ATOM_Q_START),
            int(_ATOM_Q_LEN),
            int(_ATOM_KV_BLOCKS),
            int(_ATOM_TOTAL_EXTENT),
            int(_ATOM_GLOBAL_Q_IDX),
            int(_ATOM_PTR_HIGH),
            int(_ATOM_UNUSED),
        )

        n_atoms = 0
        for i in range(int(batch_meta[int(_BATCH_N_SEQUENCES)])):
            n_tokens = int(descriptors[i, int(_SEQ_N_TOKENS)])
            cur_start_idx = int(descriptors[i, int(_SEQ_START_IDX)])
            global_start_idx = int(descriptors[i, int(_SEQ_SEEN_TOKENS)])
            remaining_toks = n_tokens

            for _ in range((n_tokens + q_block_size - 1) // q_block_size):
                atom_q_len = min(remaining_toks, q_block_size)
                end_toks = global_start_idx + atom_q_len

                atoms[n_atoms, slot_block_offset] = block_offsets[i]
                atoms[n_atoms, slot_ptr_high] = 0
                atoms[n_atoms, slot_q_start] = cur_start_idx
                atoms[n_atoms, slot_q_len] = atom_q_len
                atoms[n_atoms, slot_kv_blocks] = (
                    end_toks + kv_block_size - 1
                ) // kv_block_size
                atoms[n_atoms, slot_total_extent] = end_toks
                atoms[n_atoms, slot_global_q_idx] = global_start_idx
                atoms[n_atoms, slot_unused] = 0

                cur_start_idx += atom_q_len
                global_start_idx += atom_q_len
                remaining_toks -= atom_q_len
                n_atoms += 1

        return atoms, ragged_batch.kv_block_idx(), n_atoms


def build_blocked_flash_atoms(seq_params, q_block_size, kv_block_size, device):
    """Atoms for a batch of ``(q_len, history_len)`` sequences, and their blocks."""
    batch = RaggedBatchWrapper(seq_params, kv_block_size)

    max_atoms = sum(
        (q_len + q_block_size - 1) // q_block_size for q_len, _ in seq_params
    )
    atoms = torch.zeros(
        (max_atoms, _ATOM_SLOTS), dtype=torch.int32, device=torch.device("cpu")
    )
    atoms, kv_block_idx, n_atoms = AtomBuilder()(
        atoms, batch, q_block_size, kv_block_size
    )

    return (
        atoms[:n_atoms].to(device=device).contiguous(),
        kv_block_idx.to(device=device),
    )


_DEEPSPEED_BLOCKED_FLASH = None


def _load_deepspeed_blocked_flash():
    try:
        from deepspeed.inference.v2.inference_utils import DtypeEnum
        from deepspeed.inference.v2.kernels.ragged_ops import BlockedFlashAttn

        return BlockedFlashAttn, DtypeEnum
    except Exception as exc:
        raise RuntimeError(
            "blocked_flash baseline requires DeepSpeed's BlockedFlashAttn, "
            f"which could not be loaded: {exc!r}. Its kernel ships prebuilt in "
            "the `dskernels` package; install that for this platform."
        ) from exc


def _deepspeed_blocked_flash():
    global _DEEPSPEED_BLOCKED_FLASH
    if _DEEPSPEED_BLOCKED_FLASH is None:
        _DEEPSPEED_BLOCKED_FLASH = _load_deepspeed_blocked_flash()
    return _DEEPSPEED_BLOCKED_FLASH


def _deepspeed_atoms(seq_params, q_block_size, kv_block_size, device):
    """Atoms in the reference layout, for the DeepSpeed baseline."""
    atoms, kv_block_idx = build_blocked_flash_atoms(
        seq_params, q_block_size, kv_block_size, device
    )
    pinned = kv_block_idx.cpu().pin_memory()
    base = pinned.data_ptr()
    addresses = torch.tensor(
        [base + 4 * offset for offset in atoms[:, 0].tolist()],
        dtype=torch.int64,
        device=device,
    )
    atoms[:, 0:2] = addresses.view(torch.int32).reshape(-1, 2)
    return atoms, pinned


_deepspeed_kernels = {}


def _deepspeed_kernel(head_size, dtype):
    key = (head_size, dtype)
    if key not in _deepspeed_kernels:
        BlockedFlashAttn, DtypeEnum = _deepspeed_blocked_flash()
        _deepspeed_kernels[key] = BlockedFlashAttn(head_size, DtypeEnum(dtype))
    return _deepspeed_kernels[key]


def _deepspeed_geometry(head_size):
    from deepspeed.inference.v2.kernels.ragged_ops.blocked_flash.blocked_flash import (
        get_kv_block_size,
        get_q_block_size,
    )

    return get_q_block_size(head_size), get_kv_block_size(head_size)


def _cache_geometry(head_size):
    return _deepspeed_geometry(head_size)


def blocked_flash_input_fn(shape, dtype, device):
    """One prompt of ``context_length`` tokens against its own paged KV-cache."""
    n_tokens, head_size, n_heads_q, n_heads_kv = shape
    q_block_size, kv_block_size = _cache_geometry(head_size)

    q = torch.randn((n_tokens, n_heads_q * head_size), dtype=dtype, device=device)
    kv = torch.randn((n_tokens, 2 * n_heads_kv * head_size), dtype=dtype, device=device)
    out = torch.empty_like(q)

    n_blocks = (n_tokens + kv_block_size - 1) // kv_block_size
    padded = torch.zeros(
        (n_blocks * kv_block_size, 2 * n_heads_kv * head_size),
        dtype=dtype,
        device=device,
    )
    padded[:n_tokens] = kv
    paged = padded.reshape(n_blocks, kv_block_size, 2 * n_heads_kv * head_size)
    k_cache = (
        paged[:, :, : n_heads_kv * head_size]
        .reshape(n_blocks, kv_block_size, n_heads_kv, head_size)
        .contiguous()
    )
    v_cache = (
        paged[:, :, n_heads_kv * head_size :]
        .reshape(n_blocks, kv_block_size, n_heads_kv, head_size)
        .contiguous()
    )

    atoms, kv_block_idx = build_blocked_flash_atoms(
        [(n_tokens, 0)], q_block_size, kv_block_size, device
    )

    deepspeed_atoms, deepspeed_pinned = _deepspeed_atoms(
        [(n_tokens, 0)], q_block_size, kv_block_size, device
    )

    cu_seqlens = torch.tensor([0, n_tokens], dtype=torch.int32, device=device)
    yield (
        out,
        q,
        k_cache,
        v_cache,
        atoms,
        kv_block_idx,
        kv[:, : n_heads_kv * head_size].contiguous(),
        kv[:, n_heads_kv * head_size :].contiguous(),
        cu_seqlens,
        cu_seqlens,
        n_tokens,
        n_tokens,
        deepspeed_atoms,
        deepspeed_pinned,
        q_block_size,
    )


def torch_op(
    out,
    q,
    k,
    v,
    atoms,
    kv_block_idx,
    dense_k,
    dense_v,
    cu_q,
    cu_kv,
    max_q,
    max_kv,
    deepspeed_atoms,
    deepspeed_pinned,
    q_block_size,
):
    """DeepSpeed BlockedFlashAttn; raises when it cannot be loaded."""
    return _deepspeed_kernel(k.size(-1), out.dtype)(
        out, q, k, v, deepspeed_atoms, _SOFTMAX_SCALE
    )


def native():
    # Load eagerly so an unavailable DeepSpeed fails here instead of being
    # discovered mid-collection (and silently skipped).
    _deepspeed_blocked_flash()
    return torch_op


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    return next(iter(blocked_flash_input_fn(binding["shape"], dtype, device))), {}


def key_shape(binding):
    # First tensor: out [n_tokens, n_heads_q * head_size]; unique per shape.
    n_tokens, head_size, n_heads_q, _ = binding["shape"]
    return [n_tokens, n_heads_q * head_size]
