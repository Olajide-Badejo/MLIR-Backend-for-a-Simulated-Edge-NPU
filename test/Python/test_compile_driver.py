# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""`npu-compile`, the driver of Section 6, and its four stages.

What this file asserts is the driver's contract rather than the compiler's
arithmetic. Whether the numbers are right is `test_end_to_end.py`'s question;
whether `--emit npuisa` stops where it says it does, whether the level comes
from the compiler rather than from a list in Python, and whether a level that
does not exist is refused by name are this one's.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from npu_frontend import (
    CompileError,
    ablatable_passes,
    compile_model,
    describe_pipeline,
    generate_model,
    implemented_levels,
    run_program,
)
from npu_frontend.compile import (
    DEFAULT_SCRATCHPAD_BUDGET,
    EMIT_STAGES,
    SimulationError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "npu-compile"


@pytest.fixture(scope="module")
def lenet(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The anchor model, exported once for this file."""
    directory = tmp_path_factory.mktemp("compile-driver")
    yield generate_model("lenet", directory)


# ---------------------------------------------------------------------------
# The pipeline description, which is the compiler describing itself.
# ---------------------------------------------------------------------------


def test_the_level_table_comes_from_the_compiler() -> None:
    table = describe_pipeline()
    assert table["source"] == "lib/Pipeline/Pipeline.cpp"
    levels = {int(row["level"]): row for row in table["levels"]}
    assert sorted(levels) == [0, 1, 2]

    o0 = levels[0]
    assert o0["implemented"] is True
    assert [entry["pass"] for entry in o0["passes"]] == [
        "npu-lower-to-npuisa",
        "npu-allocate-scratchpad",
    ]

    # -O1 and -O2 are named and not built, and they name the phase that builds
    # them. A driver that could not see them at all could not tell a level that
    # is missing from one that is empty.
    for level in (1, 2):
        assert levels[level]["implemented"] is False
        assert levels[level]["arrives_at"] == "P9"
        assert levels[level]["passes"] == []


def test_only_minus_o_zero_is_implemented_at_this_phase() -> None:
    """The list is read from the compiler, not written down here.

    When P9 lands this test fails, and that is the point: the phase that adds a
    level edits the assertion in the same commit that adds it, rather than
    leaving a stale claim in a test file nobody rereads.
    """
    assert implemented_levels() == [0]


def test_the_ablatable_set_at_minus_o_zero_is_empty_and_correctly_so() -> None:
    """Both of -O0's passes are marked not ablatable, per Section 12.

    Removing either produces no program at all, so an ablation of one measures
    nothing and fails the run for a reason that has nothing to do with the
    pass. The emptiness is the right answer rather than an unfinished one.
    """
    assert ablatable_passes(0) == []


def test_an_unimplemented_level_is_refused_by_name(lenet: Path) -> None:
    with pytest.raises(CompileError) as failure:
        compile_model(lenet, level=2, emit="npu")
    message = str(failure.value)
    assert "-O2" in message
    assert "P9" in message


def test_a_level_that_is_not_a_level_lists_the_ones_that_are(lenet: Path) -> None:
    with pytest.raises(CompileError) as failure:
        compile_model(lenet, level=7, emit="npu")
    assert "-O0" in str(failure.value)


# ---------------------------------------------------------------------------
# The stages.
# ---------------------------------------------------------------------------


def test_every_stage_produces_something_and_stops_there(lenet: Path) -> None:
    for emit in EMIT_STAGES:
        result = compile_model(lenet, level=0, emit=emit)
        assert result.emit == emit
        # The stages that ran are exactly the ones up to and including this
        # one. A stage that ran and was thrown away is work the driver did for
        # nobody, and one that did not run is a stage the result cannot carry.
        expected = list(EMIT_STAGES[: EMIT_STAGES.index(emit) + 1])
        if emit == "nbin":
            # nbin is bytes and has no text stage entry.
            expected = expected[:-1]
        assert list(result.stages) == expected

        if emit == "nbin":
            assert result.binary is not None
            assert result.text is None
            assert result.binary[:4] == b"NBIN"
        else:
            assert result.text is not None
            assert result.binary is None


def test_import_and_npu_are_the_same_text_at_minus_o_zero(lenet: Path) -> None:
    """-O0 runs no `npu` level pass, so the two stages agree.

    This is a property of the level rather than of the driver, and it stops
    being true at P9 when `-O1` adds canonicalization. Asserting it now means
    the change that breaks it is the change that updates it.
    """
    imported = compile_model(lenet, level=0, emit="import")
    npu_level = compile_model(lenet, level=0, emit="npu")
    assert imported.text == npu_level.text


def test_the_npuisa_stage_is_lowered_and_allocated(lenet: Path) -> None:
    result = compile_model(lenet, level=0, emit="npuisa")
    assert result.text is not None
    assert "npuisa.scratchpad_bytes" in result.text
    assert "npuisa.dma_load" in result.text
    assert "npu.conv2d" not in result.text


def test_locations_survive_to_the_npuisa_stage(lenet: Path) -> None:
    """Section 11's NameLoc reaches the last IR stage.

    The driver asks npu-opt for debug information explicitly. Without that the
    ONNX node names are dropped at the stage boundary and the .nbin's debug
    section, which the encoder fills from these locations, would be empty for a
    reason nothing reports.
    """
    result = compile_model(lenet, level=0, emit="npuisa")
    assert result.text is not None
    assert 'loc("node_conv2d")' in result.text


def test_the_boundary_shapes_are_the_graphs_own(lenet: Path) -> None:
    result = compile_model(lenet, level=0, emit="import")
    assert result.input_shapes == ((1, 1, 28, 28),)
    assert result.output_shapes == ((1, 10),)


def test_the_budget_reaches_the_allocator(lenet: Path) -> None:
    default = compile_model(lenet, level=0, emit="npuisa")
    assert default.text is not None
    assert (
        f"npuisa.scratchpad_budget = {DEFAULT_SCRATCHPAD_BUDGET} : i64" in default.text
    )

    tighter = compile_model(lenet, level=0, emit="npuisa", budget=2 * 1024 * 1024)
    assert tighter.text is not None
    assert "npuisa.scratchpad_budget = 2097152 : i64" in tighter.text


def test_a_budget_too_small_is_a_refusal_with_the_numbers_in_it(lenet: Path) -> None:
    with pytest.raises(CompileError) as failure:
        compile_model(lenet, level=0, emit="npuisa", budget=4096)
    message = str(failure.value)
    assert "npuisa" in message
    assert "4096" in message


def test_strip_debug_produces_a_smaller_binary(lenet: Path) -> None:
    full = compile_model(lenet, level=0, emit="nbin")
    stripped = compile_model(lenet, level=0, emit="nbin", strip_debug=True)
    assert full.binary is not None and stripped.binary is not None
    assert len(stripped.binary) < len(full.binary)


def test_an_unknown_stage_lists_the_stages(lenet: Path) -> None:
    with pytest.raises(CompileError) as failure:
        compile_model(lenet, level=0, emit="assembly")
    assert "npuisa" in str(failure.value)


def test_verbose_timings_go_to_stderr(
    lenet: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A driver whose progress lands in its own output is one nobody pipes."""
    compile_model(lenet, level=0, emit="npuisa", verbose=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "npu-compile:" in captured.err
    for stage in ("import", "npu", "npuisa", "total"):
        assert stage in captured.err


# ---------------------------------------------------------------------------
# Running what was compiled.
# ---------------------------------------------------------------------------


def test_the_driver_and_the_simulator_meet(lenet: Path) -> None:
    """The walking skeleton, in three lines, with the statistics as data."""
    program = compile_model(lenet, level=0, emit="nbin")
    assert program.binary is not None
    rng = np.random.default_rng(20260831)
    x = rng.standard_normal(program.input_shapes[0]).astype(np.float32)

    answer = run_program(program.binary, [x], program.output_shapes)
    assert len(answer.outputs) == 1
    assert answer.outputs[0].shape == (1, 10)
    assert np.all(np.isfinite(answer.outputs[0]))

    # Section 10.2: this is the only instruction count in the project, it comes
    # from the simulator as a field, and nothing counts lines to get it.
    assert answer.instructions > 0
    assert answer.stats["reached_halt"] is True
    assert answer.stats["single_port"] is False


def test_the_single_port_flag_reaches_the_simulator(lenet: Path) -> None:
    program = compile_model(lenet, level=0, emit="nbin")
    assert program.binary is not None
    rng = np.random.default_rng(20260831)
    x = rng.standard_normal(program.input_shapes[0]).astype(np.float32)

    two_ports = run_program(program.binary, [x], program.output_shapes)
    one_port = run_program(program.binary, [x], program.output_shapes, single_port=True)
    assert one_port.stats["single_port"] is True
    assert one_port.stats["overlap_fraction"] == 0.0
    assert one_port.stats["cycles"] > two_ports.stats["cycles"]
    # The same program on one port computes the same answer. The flag is a
    # reproducibility switch over the cost model, not a second machine.
    np.testing.assert_array_equal(one_port.outputs[0], two_ports.outputs[0])


def test_the_wrong_number_of_inputs_is_a_refusal(lenet: Path) -> None:
    program = compile_model(lenet, level=0, emit="nbin")
    assert program.binary is not None
    with pytest.raises(SimulationError) as failure:
        run_program(program.binary, [], program.output_shapes)
    assert "npu-sim exited" in str(failure.value)


# ---------------------------------------------------------------------------
# The launcher.
# ---------------------------------------------------------------------------


def _launch(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_launcher_emits_ir_on_stdout(lenet: Path) -> None:
    completed = _launch(str(lenet), "-O", "0", "--emit", "npuisa")
    assert completed.returncode == 0, completed.stderr
    assert "npuisa.dma_load" in completed.stdout


def test_the_launcher_writes_a_binary_only_to_a_file(
    lenet: Path, tmp_path: Path
) -> None:
    """A .nbin on a terminal is a screenful somebody has to decide was not a
    diagnostic, so the driver refuses instead."""
    refused = _launch(str(lenet), "--emit", "nbin")
    assert refused.returncode == 1
    assert "-o is required" in refused.stderr

    target = tmp_path / "lenet.nbin"
    written = _launch(str(lenet), "--emit", "nbin", "-o", str(target))
    assert written.returncode == 0, written.stderr
    assert target.read_bytes()[:4] == b"NBIN"


def test_the_launcher_describes_the_pipeline() -> None:
    completed = _launch("--describe-pipeline")
    assert completed.returncode == 0, completed.stderr
    table = json.loads(completed.stdout)
    assert table["source"] == "lib/Pipeline/Pipeline.cpp"


def test_the_launcher_refuses_an_unimplemented_level(lenet: Path) -> None:
    completed = _launch(str(lenet), "-O", "2")
    assert completed.returncode == 1
    assert "P9" in completed.stderr
