# Reference as candidate (`--ref-as-candidate`)

`--ref-as-candidate` is an opt-in validation mode. It replaces the candidate
implementation with the explicit reference that the test or benchmark has
already declared, so correctness and timing exercises the reference contract
itself instead of silently measuring the candidate.

The flag is off by default and changes nothing unless it is passed.

## Correctness tests

Define a callable named `reference` in the test module and call the public
operator through its module attribute. For example, in
`tests/megatron/test_fast_gelu.py`:

```python
import pytest
import torch

import flag_train


def reference(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


@pytest.mark.fast_gelu
def test_fast_gelu():
    x = torch.arange(8, dtype=torch.float32, device=flag_train.device)
    torch.testing.assert_close(flag_train.megatron.fast_gelu(x), reference(x))
```

The plugin resolves the operator from the usual component and filename,
an existing `OP_NAME`, or a uniquely matching operator mark. Use a module-level
`OVERRIDE_KEY` containing a qualified path such as `megatron.fast_gelu` to
specify the target explicitly.

During the test call, the plugin temporarily replaces the public attribute
with the reference and restores it afterward. Access the operator through the
module when calling it so the replacement is visible. The explicit reference
must match the candidate's signature, outputs, and mutation contract.
A missing reference, ambiguous target, or test that never calls the replaced
operator fails with `ReferenceContractError`. Calls to the oracle directly and
calls made by fixtures do not count. Each selected operator needs at least one
successful candidate-side call. Individual cases for that operator may skip.

Existing `get_candidate(..., reference=...)` calls still work and emit
`DeprecationWarning`. Migrate them by declaring `reference` at module level
and calling the public operator entry point.

```sh
pytest tests -m "<operator>" --ref-as-candidate
```

## Benchmarks

The benchmark timing path uses the benchmark's explicit `torch_op` as the
candidate-side reference. It does not fall back to `train_op` or `use_train`,
and it does not resolve an eager candidate. The explicit reference is validated
before any timing, including the native baseline, so a missing or non-callable
`torch_op` reports the reference-contract error rather than a generic timer
failure. The native-baseline timing is unchanged and is not counted as
candidate execution.

Backward benchmarks keep their ordinary autograd-based timing. The selected
reference is run once to build its graph outside the timed region, and the
timed operation is `torch.autograd.grad` on that graph. Only those candidate-side
derivative calls can satisfy the reference contract: constructing the graph, or
running the baseline derivative, is not treated as candidate execution. Explicit
backward operator benchmarks (functions that take the output gradient as an
input and keep `is_backward=False`) are unaffected.

```sh
pytest benchmark -m "<operator>" --level core --ref-as-candidate --record json
```

Each measured workload must actually invoke the selected reference; otherwise
the test fails. Cases with no selected workload are reported as not selected,
not as successful reference timings. JSON/log records produced by this mode
include `candidate_source: "reference"` so the timings are not mistaken for
optimized candidate performance.

## Rejected combinations

`--ref-as-candidate` is a focused validation mode, so it rejects options that
change what is being validated: operator overrides, dummy/report-only modes,
case listing, query, collect-only/collect-marks, and parallel/xdist execution.
A clear `pytest.UsageError` is emitted. Ordinary test/case selection and
`-k`-style node filtering remain available.
