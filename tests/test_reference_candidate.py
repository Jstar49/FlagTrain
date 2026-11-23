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
"""Regression tests for reference-as-candidate selection and timing."""

from types import SimpleNamespace

import pytest
import torch

import flag_train
from flag_train.reference_candidate import (
    ReferenceContractError,
    get_candidate,
    reference_test_mode,
)

pytest_plugins = ["pytester"]


def test_candidate_defaults_to_live_operator(monkeypatch):
    candidate = lambda: "candidate"
    monkeypatch.setattr(flag_train, "megatron", SimpleNamespace(probe=candidate))
    with pytest.warns(
        DeprecationWarning, match="define a callable.*reference"
    ) as warnings:
        assert (
            get_candidate("megatron.probe", reference=lambda: "reference") is candidate
        )
    assert warnings[0].filename == __file__


def test_reference_mode_selects_explicit_reference():
    with reference_test_mode(True) as state:
        with pytest.warns(
            DeprecationWarning, match="public operator entry point"
        ) as warnings:
            candidate = get_candidate("megatron.missing", reference=lambda: "reference")
        assert warnings[0].filename == __file__
        assert candidate() == "reference"
        assert state.candidate_calls == state.reference_calls == 1


@pytest.mark.parametrize("reference", [None, 42])
def test_reference_mode_rejects_invalid_reference(reference):
    with reference_test_mode(True):
        with pytest.raises(ReferenceContractError, match="No predefined reference"):
            get_candidate("megatron.missing", reference=reference)


@pytest.mark.parametrize("run_derivative", [True, False])
def test_reference_backward_requires_candidate_derivative(monkeypatch, run_derivative):
    from benchmark import base
    from benchmark.consts import BenchMode

    bench = base.Benchmark("probe", torch.square, is_backward=True)
    bench.to_bench_metrics = ["latency_base", "latency"]
    monkeypatch.setattr(
        base,
        "Config",
        SimpleNamespace(
            mode=BenchMode.CUDAGRAPH, skip_native=False, repetition=1, base_data=None
        ),
    )
    timer_calls = iter([True, run_derivative])

    def timer(function, **kwargs):
        if next(timer_calls):
            function()
        return 1.0

    monkeypatch.setattr(base.triton.testing, "do_bench_cudagraph", timer)
    value = torch.tensor(2.0, requires_grad=True)

    with reference_test_mode(True) as state:
        if run_derivative:
            metric = bench._measure_input((value,))
            assert metric.candidate_source == "reference"
            assert state.candidate_derivative_calls == state.completed_timings == 1
        else:
            # The baseline derivative and candidate graph both ran.
            with pytest.raises(
                pytest.fail.Exception, match="candidate-side derivative"
            ):
                bench._measure_input((value,))
            assert state.candidate_calls == 1
            assert state.candidate_derivative_calls == state.completed_timings == 0


@pytest.fixture
def automatic_reference_suite(pytester, monkeypatch):
    def original(value):
        return value + 10

    monkeypatch.setattr(flag_train.megatron, "reference_probe", original, raising=False)
    monkeypatch.setattr(
        flag_train.megatron,
        "__all__",
        [*flag_train.megatron.__all__, "reference_probe"],
    )
    pytester.makeconftest("""
from flag_train.reference_candidate import (
    add_reference_candidate_argument as pytest_addoption,
    configure_reference_candidate,
)

def pytest_configure(config):
    config.addinivalue_line("markers", "reference_probe: declare the probe operator")
    configure_reference_candidate(config)
""")
    return pytester, original


def test_module_reference_replaces_marked_public_call(automatic_reference_suite):
    suite, original = automatic_reference_suite
    suite.makepyfile("""
import pytest

import flag_train


def reference(value):
    return value + 1


@pytest.mark.reference_probe
def test_probe(request):
    actual = flag_train.megatron.reference_probe(2)
    assert actual == (reference(2) if request.config.getoption("ref_as_candidate") else 12)
""")
    suite.runpytest_inprocess().assert_outcomes(passed=1)
    suite.runpytest_inprocess("--ref-as-candidate").assert_outcomes(passed=1)
    assert flag_train.megatron.reference_probe is original


def test_conventional_component_file_can_replace_missing_operator(
    automatic_reference_suite,
):
    suite, _ = automatic_reference_suite
    component = suite.path / "tests" / "megatron"
    component.mkdir(parents=True)
    operator = "auto_reference_unimplemented_probe"
    (component / f"test_{operator}.py").write_text(f"""
import flag_train


def reference(value):
    return value + 1


def test_probe():
    assert flag_train.megatron.{operator}(2) == 3
""")
    suite.runpytest_inprocess("--ref-as-candidate").assert_outcomes(passed=1)
    assert not hasattr(flag_train.megatron, operator)


@pytest.mark.parametrize(
    "declaration,call,diagnostic",
    [
        ("", "flag_train.megatron.reference_probe(2)", "No predefined reference"),
        (
            "reference = None",
            "flag_train.megatron.reference_probe(2)",
            "No predefined reference",
        ),
        (
            "def reference(value): return value + 1",
            "reference(2)",
            "did not complete a candidate-side call",
        ),
    ],
)
def test_automatic_reference_contract_failures(
    automatic_reference_suite, declaration, call, diagnostic
):
    suite, original = automatic_reference_suite
    suite.makepyfile(test_reference_probe=f"""
import flag_train

{declaration}

def test_probe():
    {call}
""")
    result = suite.runpytest_inprocess("--ref-as-candidate")
    result.assert_outcomes(failed=1)
    assert diagnostic in result.stdout.str()
    assert flag_train.megatron.reference_probe is original


def test_legacy_reference_without_module_declaration_still_works(
    automatic_reference_suite,
):
    suite, original = automatic_reference_suite
    suite.makepyfile(test_reference_probe="""
from flag_train.reference_candidate import get_candidate


def test_probe():
    candidate = get_candidate("megatron.reference_probe", reference=lambda value: value + 1)
    assert candidate(2) == 3
""")
    suite.runpytest_inprocess("--ref-as-candidate").assert_outcomes(passed=1)
    assert flag_train.megatron.reference_probe is original


def test_bad_explicit_reference_target_fails_the_case(automatic_reference_suite):
    suite, _ = automatic_reference_suite
    suite.makepyfile("""
OVERRIDE_KEY = "missing.operator"


def reference(value):
    return value


def test_probe():
    raise AssertionError("invalid reference target ran the test body")
""")
    result = suite.runpytest_inprocess("--ref-as-candidate")
    result.assert_outcomes(failed=1)
    assert "ReferenceContractError" in result.stdout.str()
    assert "Unsupported operator namespace" in result.stdout.str()
    assert "invalid reference target ran the test body" not in result.stdout.str()


def test_legacy_reference_without_inferred_target_still_works(
    automatic_reference_suite,
):
    suite, _ = automatic_reference_suite
    suite.makepyfile("""
from flag_train.reference_candidate import get_candidate


def reference(value):
    return value + 1


def test_probe():
    candidate = get_candidate("megatron.reference_probe", reference=reference)
    assert candidate(2) == 3
""")
    suite.runpytest_inprocess("--ref-as-candidate").assert_outcomes(passed=1)


@pytest.mark.parametrize("skip_second", [False, True])
def test_reference_requires_a_call_for_each_selected_operator(
    automatic_reference_suite, monkeypatch, skip_second
):
    suite, original = automatic_reference_suite
    monkeypatch.setattr(
        flag_train.megatron, "reference_other_probe", original, raising=False
    )
    monkeypatch.setattr(
        flag_train.megatron,
        "__all__",
        [*flag_train.megatron.__all__, "reference_other_probe"],
    )
    suite.makeini(
        "[pytest]\nmarkers =\n    reference_other_probe: declare another operator"
    )
    suite.makepyfile(test_reference_probe=f"""
import pytest

import flag_train


def reference(value):
    return value + 1


@pytest.mark.reference_probe
def test_called():
    assert flag_train.megatron.reference_probe(2) == 3


@pytest.mark.reference_probe
@pytest.mark.skip(reason="individual case is unsupported")
def test_skipped_case():
    raise AssertionError("setup did not skip the case")


@pytest.mark.reference_other_probe
@pytest.mark.skipif({skip_second}, reason="operator is unsupported")
def test_distinct_operator():
    assert flag_train.megatron.reference_other_probe(2) == 3
""")
    result = suite.runpytest_inprocess(
        "-m", "reference_probe or reference_other_probe", "--ref-as-candidate"
    )
    if skip_second:
        result.assert_outcomes(passed=1, skipped=2)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert "flag_train.megatron.reference_other_probe" in result.stdout.str()
    else:
        result.assert_outcomes(passed=2, skipped=1)
        assert result.ret == pytest.ExitCode.OK


@pytest.mark.parametrize("prefix", ["megatron", "flag_train.megatron"])
def test_qualified_op_name_with_automatic_and_legacy_reference(
    automatic_reference_suite, prefix
):
    suite, original = automatic_reference_suite
    component = suite.path / "tests" / "megatron"
    component.mkdir(parents=True)
    (component / "test_qualified_reference.py").write_text(f"""
import pytest

import flag_train
from flag_train.reference_candidate import get_candidate

OP_NAME = "{prefix}.reference_probe"


def reference(value):
    return value + 1


def test_auto():
    assert flag_train.megatron.reference_probe(2) == 3


def test_legacy():
    with pytest.warns(DeprecationWarning):
        candidate = get_candidate(OP_NAME, reference=reference)
    assert candidate(2) == 3
""")
    suite.runpytest_inprocess("--ref-as-candidate").assert_outcomes(passed=2)
    assert flag_train.megatron.reference_probe is original
