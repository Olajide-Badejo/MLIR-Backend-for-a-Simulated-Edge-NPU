# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The roofline of Section 16.6, and the walk every per layer number comes from.

Two things are being held here and they are different in kind.

The first is arithmetic: the bound is `max(bytes / bw, effective_macs / peak)`,
the branch that produced it is recorded, and a charge below it is `below_bound`
rather than rounded up to a pass. Those are checked against numbers a reader can
do in their head.

The second is the honest limit of the check, and it is asserted rather than only
written down. Under the cost model of Section 5.5 `effective_macs` is *defined*
as `cycles * peak`, so the compute branch of the roofline is identically the
kernel's own cycle count; and the DMA charge is bytes over bandwidth plus a
descriptor, so the memory branch is identically below the transfer's charge. The
roofline therefore cannot fail against this cost model as it stands, and
`test_the_compute_branch_is_the_charge_itself` and
`test_the_memory_branch_is_below_what_the_transfer_costs` are that statement in a
form that will start failing the day it stops being true. That is the point: the
check earns its place as a **regression** bound, against the phase that
introduces a charge which is no longer built from the traffic, not as evidence
that today's numbers are right.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from npu_frontend import cost_model, npuisa_walk
from npu_frontend.results import RESULTS_DIR, load_result

from tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import roofline  # noqa: E402

PEAK = cost_model.PEAK_MACS_PER_CYCLE_F32
BANDWIDTH = cost_model.DRAM_BANDWIDTH_BYTES_PER_CYCLE


# ---------------------------------------------------------------------------
# The bound.
# ---------------------------------------------------------------------------


def test_the_bound_is_the_later_of_the_two_branches() -> None:
    """Section 16.6 writes it as a `min` in the denominator; that is a `max` here.

    `effective_macs / min(intensity * bw, peak)` and
    `max(bytes / bw, effective_macs / peak)` are the same number, and this test
    exists because the two forms look nothing alike and a reader checking one
    against the other should not have to take it on faith.
    """
    effective_macs = 1_000_000.0
    bytes_moved = 64_000
    intensity = effective_macs / bytes_moved
    as_written = effective_macs / min(intensity * BANDWIDTH, PEAK)

    bound = roofline.bound_for(
        effective_macs=effective_macs,
        dram_bytes=bytes_moved,
        charged_cycles=as_written,
        peak=PEAK,
    )
    assert bound.bound_cycles == pytest.approx(as_written, rel=1e-12)


def test_a_memory_bound_layer_records_which_branch_bound_it() -> None:
    """A layer that moves many bytes for few multiplies is bound by the memory
    branch, and Section 16.6 asks for that to be recorded rather than inferred."""
    bound = roofline.bound_for(
        effective_macs=16.0, dram_bytes=160_000, charged_cycles=10_000.0, peak=PEAK
    )
    assert bound.bound_by == roofline.MEMORY_BRANCH
    assert bound.memory_branch_cycles == 160_000 / BANDWIDTH
    assert bound.verdict == roofline.AT_OR_ABOVE


def test_a_charge_under_the_memory_branch_is_below_bound() -> None:
    """The failure the check exists for, proved rather than assumed.

    Section 19.1's rule applied to an assertion instead of to a CI step: a check
    nobody has seen go red is a check nobody knows works. 160000 bytes take 10000
    cycles to move at 16 bytes per cycle, so a charge of 9999 asserts the
    interconnect moved them faster than it can.
    """
    bound = roofline.bound_for(
        effective_macs=16.0, dram_bytes=160_000, charged_cycles=9_999.0, peak=PEAK
    )
    assert bound.verdict == roofline.BELOW
    assert bound.headroom < 0.0


def test_a_layer_that_moved_no_bytes_records_no_intensity() -> None:
    """`operational_intensity` is null rather than infinite, and that is a real
    state: the allocator stages a convolution's operands into the scratchpad, so
    a layer whose operands were staged for an earlier layer touches DRAM not at
    all. A division by zero recorded as a large number would be an intensity this
    project did not measure appearing in a plot as if it had."""
    bound = roofline.bound_for(
        effective_macs=4096.0, dram_bytes=0, charged_cycles=16.0, peak=PEAK
    )
    assert bound.operational_intensity is None
    assert bound.bound_by == roofline.COMPUTE_BRANCH


# ---------------------------------------------------------------------------
# What the check is worth, asserted rather than described.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "reduction", "columns"),
    [(1, 400, 120), (4, 40, 33), (64, 16, 16), (256, 1, 1)],
)
def test_the_compute_branch_is_the_charge_itself(
    rows: int, reduction: int, columns: int
) -> None:
    """The compute branch is a tautology under this cost model, and this says so.

    `gemm_charge` sets `effective_macs = cycles * peak`, so
    `effective_macs / peak` is `cycles` exactly. Section 16.6 warns that the
    compute branch is partly the cost model grading its own homework; the
    measurement is that it is entirely so, and `docs/NUMBERS.md` records what
    that leaves the check worth.

    **The day this test fails is the day the compute branch starts meaning
    something**, because it would mean `effective_macs` had stopped being defined
    from the cycle count. That is a change worth noticing, which is why the
    tautology is asserted instead of only being written in a docstring.
    """
    charge = cost_model.gemm_charge(rows, reduction, columns, PEAK)
    assert charge.effective_macs / PEAK == pytest.approx(charge.cycles, rel=1e-15)


def test_the_memory_branch_is_below_what_the_transfer_costs() -> None:
    """And so is the memory branch, for a different reason: the descriptor.

    `dma_cycles` charges bytes over bandwidth **plus** a fixed per descriptor
    cost, so a transfer always costs strictly more than the memory branch of the
    bound its bytes produce. The gap is the descriptor, and it is what the
    tightest headroom in the suite is made of.
    """
    for byte_count in (64, 4096, 192_480):
        elements = byte_count // 4
        charged = cost_model.dma_cycles(byte_count, elements, 1)
        assert charged > byte_count / BANDWIDTH
        assert charged - byte_count / BANDWIDTH == cost_model.DMA_DESCRIPTOR_CYCLES


# ---------------------------------------------------------------------------
# The walk, against the machine.
# ---------------------------------------------------------------------------


def test_the_walker_refuses_an_operation_it_does_not_know() -> None:
    """An unknown operation raises and names itself.

    A walk that skipped what it did not understand would report a MAC total
    quietly short of the machine's, which is Section 16.4's rule that an absent
    number must never be indistinguishable from a zero, written for a walker.
    """
    text = (
        "module {\n"
        "  func.func @main() {\n"
        "    npuisa.sorcery ins(%a : memref<4xf32, #npu.scratchpad>) "
        "outs(%b : memref<4xf32, #npu.scratchpad>) loc(#loc)\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(npuisa_walk.WalkError, match="npuisa.sorcery"):
        npuisa_walk.walk(text)


def test_the_walker_reads_the_asynchronous_transfer_forms() -> None:
    """D-0058, and the reason it is a test rather than a line in a set.

    `-npu-double-buffer` emits `npuisa.dma_load_async` with an `npuisa.await`
    where the load was, and until the pass started firing no compiled program
    contained either. The walk refused the first one it saw. The charge is the
    synchronous charge, because the encoder emits `DMA_LOAD` for both forms, and
    the await is skipped rather than charged zero, because the walk's positions
    are the machine's instruction indices and an operation charged zero still
    occupies one.
    """
    text = (
        "module {\n"
        "  func.func @main() {\n"
        "    %0 = npuisa.dma_load_async %cst, %view : "
        "memref<4xf32, #npu.dram> to memref<4xf32, #npu.scratchpad> loc(#loc)\n"
        "    npuisa.relu ins(%a : memref<4xf32, #npu.scratchpad>) "
        "outs(%b : memref<4xf32, #npu.scratchpad>) loc(#loc)\n"
        "    npuisa.await %0 loc(#loc)\n"
        "  }\n"
        "}\n"
    )
    operations = npuisa_walk.walk(text)

    assert [operation.op for operation in operations] == ["dma_load_async", "relu"]
    assert "await" in npuisa_walk.DECLARATION_OPS
    assert "dma_load_async" in npuisa_walk.TRANSFER_OPS


def test_an_empty_walk_is_refused() -> None:
    """An empty table passing every assertion is how a check stops checking."""
    with pytest.raises(npuisa_walk.WalkError, match="no npuisa operations"):
        npuisa_walk.walk("module {\n  func.func @main() {\n  }\n}\n")


@pytest.mark.parametrize(
    "cell",
    [
        "lenet-O2-default-n1-fp32-normal",
        "depthwise_separable-O2-default-n1-fp32-normal",
        "inception_block-O2-default-n1-fp32-normal",
        "dilated_stack-O2-default-n1-fp32-normal",
    ],
)
def test_the_walk_reproduces_the_machines_totals(cell: str) -> None:
    """The walk agrees with the simulator about the counted quantities, exactly.

    Four models rather than one, chosen for the shapes that make the walk hard:
    `depthwise_separable` for grouping, `inception_block` for the concatenation
    and the 1 by 1 convolutions, `dilated_stack` for dilation and a transpose,
    `lenet` for the pooling and the three matmuls.
    """
    tool("npu-opt")
    path = RESULTS_DIR / f"{cell}.json"
    if not path.is_file():
        pytest.skip(f"{cell} is not a committed cell in this checkout")
    result = load_result(path)

    with tempfile.TemporaryDirectory(prefix="npu-roofline-test-") as directory:
        text = roofline.Compiler(Path(directory)).allocated_ir(result)
    operations = npuisa_walk.attribute_transfers(npuisa_walk.walk(text), text)

    # Raises on a disagreement, naming every field that disagreed.
    npuisa_walk.check_against_result(operations, result)

    walked = npuisa_walk.totals(operations)
    assert walked.unattributed_dram_bytes == 0, (
        "every transfer in this suite exists to feed a layer or to drain one, so "
        "a byte charged to no layer means the attribution missed an alias. It "
        "missed the ADR 0005 broadcast casts once already."
    )
    assert walked.layers > 0


def test_every_layer_of_a_real_cell_is_at_or_above_its_bound() -> None:
    """One cell through the whole path, end to end.

    The suite wide answer is `experiments/roofline.py`'s own exit code and is run
    in the verification matrix rather than here, because compiling 175 cells is
    not a unit test. This asserts the path works on one.
    """
    tool("npu-opt")
    path = RESULTS_DIR / "lenet-O2-default-n1-fp32-normal.json"
    if not path.is_file():
        pytest.skip("lenet-O2-default-n1-fp32-normal is not committed here")
    result = load_result(path)

    with tempfile.TemporaryDirectory(prefix="npu-roofline-test-") as directory:
        text = roofline.Compiler(Path(directory)).allocated_ir(result)
    answer = roofline.roofline_for(result, text)

    assert not answer.violations
    assert answer.whole.verdict == roofline.AT_OR_ABOVE
    assert all(layer.bound.verdict == roofline.AT_OR_ABOVE for layer in answer.layers)
    # `lenet` has two convolutions and three matmuls and nothing else with a MAC
    # in it. A walk that had folded two of them together would still pass every
    # bound above, and would be wrong.
    assert len(answer.layers) == 5


def test_the_schema_block_carries_no_null_reason() -> None:
    """A field this phase fills must not also carry the reason it was empty.

    `validate_result` refuses a field carrying both, so a half done fill is red.
    This asserts the producing side never emits the pair in the first place,
    which is the cheaper place to find it.
    """
    tool("npu-opt")
    path = RESULTS_DIR / "resnet_block-O2-default-n1-fp32-normal.json"
    if not path.is_file():
        pytest.skip("resnet_block-O2-default-n1-fp32-normal is not committed here")
    result = load_result(path)

    with tempfile.TemporaryDirectory(prefix="npu-roofline-test-") as directory:
        text = roofline.Compiler(Path(directory)).allocated_ir(result)
    block = roofline.roofline_for(result, text).as_schema()

    assert set(block) == {
        "roofline_bound_cycles",
        "roofline_bound_cycles_per_layer",
        "operational_intensity",
        "roofline_verdict",
    }
    assert not any(key.endswith("_null_reason") for key in block)
    assert all(value is not None for value in block.values())
    assert block["roofline_verdict"] in {roofline.AT_OR_ABOVE, roofline.BELOW}


def quantized_cell(model: str, level: int) -> tuple[dict, str]:
    """A quantized program and the result fields the walk is checked against.

    Compiled from the model's committed profile and run on the machine, with
    the simulator's own counts in the shape a result cell carries them.
    """
    from npu_frontend import compile_model, generate_model, run_program
    from npu_frontend.input_classes import make_inputs

    profile = REPO_ROOT / "experiments" / "calibration" / f"{model}.json"
    with tempfile.TemporaryDirectory(prefix="npu-int8-walk-") as directory:
        onnx = generate_model(model, directory)
        program = compile_model(onnx, level=level, emit="nbin", calibrate=str(profile))
    assert program.binary is not None
    inputs = make_inputs("normal", program.input_shapes, model=model, batch=1)
    stats = run_program(program.binary, inputs, program.output_shapes).stats
    result = {
        "cell": {
            "name": f"{model}-O{level}-default-n1-int8-normal",
            "quantized": True,
            "scratchpad_budget_bytes": 1048576,
        },
        "instruction_count": stats["instructions"],
        "simulation": {
            "macs": stats["macs"],
            "int8_macs": stats["int8_macs"],
            "effective_macs": stats["effective_macs"],
            "utilization": stats["utilization"],
            "delta": stats["delta"],
            "dram_bytes_read": stats["dram_bytes_read"],
            "dram_bytes_written": stats["dram_bytes_written"],
            "dram_bytes_total": stats["dram_bytes_read"] + stats["dram_bytes_written"],
            "simulated_cycles": stats["cycles"],
            "dma_cycles": stats["dma_cycles"],
            "compute_cycles": stats["compute_cycles"],
        },
    }
    return result, program.stages["npuisa"]


def test_an_int8_program_is_walked_and_bounded_at_its_own_peak() -> None:
    """*Added at P14.* The walk reads a quantized program and agrees with the
    machine on it, and each contraction is bounded at the peak it runs at.

    The walker read f32 alone until the quantized cells. Here it reads i8
    activations and weights, an i32 bias and rescale table, and the `QUANT` and
    `DEQUANT` on either side, and reproduces the machine's counts exactly and
    its two timelines to the cycle. An integer contraction is charged at the
    int8 peak, so its compute branch is its effective MACs over that peak; at
    the f32 peak every compute bound int8 layer would read four times below a
    bound it never claimed to meet.
    """
    tool("npu-opt")
    result, text = quantized_cell("lenet", 0)
    operations = npuisa_walk.attribute_transfers(npuisa_walk.walk(text), text)
    npuisa_walk.check_against_result(operations, result)
    walked = npuisa_walk.totals(operations)
    assert walked.compute_cycles == result["simulation"]["compute_cycles"]
    assert walked.dma_cycles == result["simulation"]["dma_cycles"]

    contractions = [operation for operation in operations if operation.is_compute]
    assert contractions and all(operation.is_integer for operation in contractions)
    assert {operation.peak for operation in contractions} == {
        cost_model.PEAK_MACS_PER_CYCLE_I8
    }
    assert {"quant", "dequant"} <= {operation.op for operation in operations}

    answer = roofline.roofline_for(result, text)
    assert answer.violations == []
    assert answer.whole.verdict == roofline.AT_OR_ABOVE


def test_a_program_mixing_int8_and_f32_macs_is_refused_a_whole_bound() -> None:
    """A whole program bound needs one peak, and a mixed program has none.

    No cell of this suite mixes the two, because every contraction of a
    quantized model contracts. The refusal is what keeps a future partial
    calibration from being bounded at either peak and called a measurement.
    """
    tool("npu-opt")
    result, text = quantized_cell("lenet", 0)
    result["simulation"]["int8_macs"] = result["simulation"]["macs"] - 1
    with pytest.raises(roofline.RooflineError, match="no single peak"):
        roofline.roofline_for(result, text)


def test_a_committed_quantized_cell_is_recompiled_from_its_profile() -> None:
    """*D-0071.* The analysis recompiles a quantized cell as the program it was.

    A result file carries the numbers and not the program, so the roofline and
    the SCALE-Sim export recompile each cell. They did so from the model, batch,
    level, budget and ablation, which a quantized cell shares with its fp32
    twin, so every committed quantized cell was walked as its fp32 twin and the
    walk's check refused it. The quantized program is the one the walk must see,
    and the fp32 twin in the same compiler must still be the fp32 program.
    """
    tool("npu-opt")
    quantized_path = RESULTS_DIR / "lenet-O2-default-n1-int8-normal.json"
    twin_path = RESULTS_DIR / "lenet-O2-default-n1-fp32-normal.json"
    if not (quantized_path.is_file() and twin_path.is_file()):
        pytest.skip("the lenet -O2 pair is not committed in this checkout")
    quantized = load_result(quantized_path)
    twin = load_result(twin_path)

    with tempfile.TemporaryDirectory(prefix="npu-roofline-test-") as directory:
        compiler = roofline.Compiler(Path(directory))
        quantized_text = compiler.allocated_ir(quantized)
        twin_text = compiler.allocated_ir(twin)

    assert quantized_text != twin_text
    for result, text in ((quantized, quantized_text), (twin, twin_text)):
        operations = npuisa_walk.attribute_transfers(npuisa_walk.walk(text), text)
        npuisa_walk.check_against_result(operations, result)
    answer = roofline.roofline_for(quantized, quantized_text)
    assert answer.violations == []
