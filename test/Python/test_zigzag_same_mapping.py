# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The ZigZag export of Section 16.5, and the one thing it has to get right.

The comparison this module supports is only worth anything if ZigZag evaluates
the loop nest the compiler chose rather than one of its own, so the tests here
are mostly about that single property approached from four sides: the mapping
attribute is read field by field and a missing field is a refusal, every
dimension's factors multiply out to its size, an ordering that does not is
refused before ZigZag ever sees it, and the end to end case reads the nest back
off ZigZag's own evaluation and compares it to what was handed over.

**The within tile order has a test of its own and it is the reason it does.**
The first version of this export put the reduction innermost, which describes a
machine that reloads its weights per output position. This one does not:
`cost_model.gemm_charge` streams the activation rows through a loaded array, so
the rows are the innermost loop and the reduction bands are the outer one.
ZigZag's own search is what found the mistake, by preferring a nest 57 percent
cheaper that turned out to be this machine's actual dataflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from npu_frontend.model_generator import generate_model

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import zigzag_same_mapping as export  # noqa: E402

#: One convolution and one matmul as the compiler prints them, trimmed to the
#: two operations and their locations. Copied out of a real `--emit npu` run
#: rather than composed, so a printer change breaks this rather than passing.
TILED_IR = """#loc = loc(unknown)
#loc5 = loc("node_conv2d_1")
#loc8 = loc("node_linear")
module {
  func.func @main() {
    %4 = npu.conv2d ins(%a, %b, %c : tensor<1x8x5x8xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>) outs(%d : tensor<1x8x4x8xf32>) {dilations = array<i64: 1, 1>, npu.tiling_choice = {loop_order = "domain", makespan_cycles = 1.400000e+03 : f64, spatial_factors = array<i64: 16, 16>, strategy = "exhaustive", temporal_tiles = array<i64: 1, 1, 8, 4, 8>, tile_bytes = 6432 : i64, tile_count = 2 : i64}, pads = array<i64: 1, 1, 0, 1>, strides = array<i64: 1, 1>} -> tensor<1x8x4x8xf32> loc(#loc5)
    %14 = npu.matmul ins(%e, %f, %g : tensor<1x400xf32>, tensor<400x30xf32>, tensor<30xf32>) outs(%h : tensor<1x30xf32>) {npu.tiling_choice = {loop_order = "domain", makespan_cycles = 1.440200e+04 : f64, spatial_factors = array<i64: 16, 16>, strategy = "exhaustive", temporal_tiles = array<i64: 1, 1, 30, 1, 1>, tile_bytes = 51440 : i64, tile_count = 4 : i64}} -> tensor<1x30xf32> loc(#loc8)
    return
  }
}
"""

#: The same two layers untiled, which is where their own shapes come from.
UNTILED_IR = """#loc = loc(unknown)
#loc5 = loc("node_conv2d_1")
#loc8 = loc("node_linear")
module {
  func.func @main() {
    %2 = npu.conv2d ins(%a, %b, %c : tensor<1x8x8x8xf32>, tensor<8x8x3x3xf32>, tensor<8xf32>) outs(%d : tensor<1x8x8x8xf32>) {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>, strides = array<i64: 1, 1>} -> tensor<1x8x8x8xf32> loc(#loc5)
    %12 = npu.matmul ins(%e, %f, %g : tensor<1x400xf32>, tensor<400x120xf32>, tensor<120xf32>) outs(%h : tensor<1x120xf32>) -> tensor<1x120xf32> loc(#loc8)
    return
  }
}
"""


def conv_pair() -> tuple[export.Problem, export.Choice]:
    return (
        export.problems(UNTILED_IR)["node_conv2d_1"],
        export._choices(TILED_IR)["node_conv2d_1"],
    )


def matmul_pair() -> tuple[export.Problem, export.Choice]:
    return (
        export.problems(UNTILED_IR)["node_linear"],
        export._choices(TILED_IR)["node_linear"],
    )


def test_the_shapes_are_read_off_the_types_the_compiler_prints() -> None:
    found = export.problems(UNTILED_IR)
    convolution = found["node_conv2d_1"]
    assert convolution.kind == "conv2d"
    assert convolution.sizes == {
        "B": 1,
        "K": 8,
        "G": 1,
        "OY": 8,
        "OX": 8,
        "C": 8,
        "FY": 3,
        "FX": 3,
    }
    assert convolution.pads == (1, 1, 1, 1)
    assert convolution.strides == (1, 1)
    assert convolution.input_height == 8

    matmul = found["node_linear"]
    assert matmul.kind == "matmul"
    assert matmul.sizes == {"C": 400, "D": 1, "K": 120}


def test_the_location_is_what_joins_a_tiled_layer_to_its_own_shape() -> None:
    """A tiled operation carries the tile's shape and the original layer's name.

    The join is the whole reason the export compiles the model twice, and it
    holds because MLIR propagates the location onto the operations a pattern
    builds. Asserted rather than assumed: a tiled convolution's own operand type
    is `1x8x5x8` and the layer it came from is `1x8x8x8`, so reading the shape
    off the tiled operation would export a mapping for a layer that does not
    exist.
    """
    tiled = export.problems(TILED_IR)["node_conv2d_1"]
    untiled = export.problems(UNTILED_IR)["node_conv2d_1"]
    assert tiled.sizes["OY"] == 4
    assert untiled.sizes["OY"] == 8
    assert set(export._choices(TILED_IR)) == {"node_conv2d_1", "node_linear"}
    assert set(export._choices(TILED_IR)) <= set(export.problems(UNTILED_IR))


def test_the_mapping_attribute_is_read_field_by_field() -> None:
    choice = export._choices(TILED_IR)["node_conv2d_1"]
    assert choice.temporal_tiles == (1, 1, 8, 4, 8)
    assert choice.spatial_factors == (16, 16)
    assert choice.tile_count == 2
    assert choice.tile_bytes == 6432
    assert choice.makespan_cycles == pytest.approx(1400.0)
    assert choice.strategy == "exhaustive"
    assert choice.loop_order == "domain"


def test_a_mapping_attribute_that_lost_a_field_is_refused() -> None:
    """The attribute changing shape is a refusal and not a default.

    An exporter that filled a missing `makespan_cycles` with a zero would report
    a comparison against nothing and call it agreement.
    """
    broken = TILED_IR.replace("makespan_cycles = 1.400000e+03 : f64, ", "")
    with pytest.raises(export.ExportError, match="missing"):
        export._choices(broken)


def test_the_spatial_fill_is_the_largest_product_of_divisors() -> None:
    assert export.fill_within([("C", 16)], 16) == [("C", 16)]
    assert export.fill_within([("C", 8)], 16) == [("C", 8)]
    assert export.fill_within([("C", 400)], 16) == [("C", 16)]


def test_a_reduction_that_does_not_factorise_fills_less_than_the_array() -> None:
    """Eight channels of a three by three kernel is 72 and 72 has no factor 16.

    This project folds that reduction into bands of sixteen and pays for the
    part band; ZigZag unrolls a divisor. Twelve is the best that can be done and
    the four rows that go missing are the named approximation of this export
    rather than a discrepancy in it.
    """
    chosen = export.fill_within([("C", 8), ("FY", 3), ("FX", 3)], 16)
    product = 1
    for _, factor in chosen:
        product *= factor
    assert product == 12
    assert product < export.ARRAY_DIM


def test_every_dimension_of_the_exported_mapping_multiplies_out() -> None:
    problem, choice = conv_pair()
    export.check_complete(problem, export.conv_mapping(problem, choice))
    problem, choice = matmul_pair()
    export.check_complete(problem, export.matmul_mapping(problem, choice))


def test_an_ordering_that_does_not_multiply_out_is_refused() -> None:
    """Which is what stops the comparison becoming the one Section 16.5 forbids.

    ZigZag treats an incomplete ordering as a hint and runs its own search, so
    an export that dropped a factor would silently compare this compiler's
    mapping against ZigZag's best. The refusal is here rather than left to
    ZigZag because ZigZag does not consider it an error.
    """
    problem, choice = conv_pair()
    mapping = export.conv_mapping(problem, choice)
    mapping.temporal = [(name, factor) for name, factor in mapping.temporal[1:]]
    with pytest.raises(export.ExportError, match="does not multiply out|covers"):
        export.check_complete(problem, mapping)


def test_the_rows_stream_innermost_because_the_array_is_weight_stationary() -> None:
    """The within tile order is the hardware's and not the pass's.

    `cost_model.gemm_charge` walks the reduction in bands, the columns in bands,
    and streams every activation row through each loaded band, so the output
    positions are the innermost temporal loop and the reduction is the outermost
    one inside a tile. An export that inverted them would describe an array that
    reloads its weights per output position, which is a different machine.
    """
    problem, choice = conv_pair()
    mapping = export.conv_mapping(problem, choice)
    names = [name for name, _ in mapping.temporal[: mapping.tile_boundary]]
    assert names.index("OX") < names.index("C")
    assert names.index("OY") < names.index("FX")
    assert names[0] == "OX"

    problem, choice = matmul_pair()
    mapping = export.matmul_mapping(problem, choice)
    names = [name for name, _ in mapping.temporal[: mapping.tile_boundary]]
    assert names == ["D", "K", "C"]


def test_the_tile_loops_are_the_outer_half_and_the_boundary_says_where() -> None:
    problem, choice = conv_pair()
    mapping = export.conv_mapping(problem, choice)
    tiles = dict(mapping.temporal[mapping.tile_boundary :])
    assert tiles["OY"] == 2
    assert tiles["OX"] == 1
    assert tiles["K"] == 1
    assert choice.tile_count == 2


def test_the_accelerator_is_this_projects_own_constants() -> None:
    """Nothing in the accelerator description is chosen for this comparison."""
    text = export.accelerator_text(6464)
    assert f"sizes: [{export.ARRAY_DIM}, {export.ARRAY_DIM}]" in text
    assert f"size: {6464 * 8}" in text
    assert f"bandwidth_max: {export.DRAM_BITS_PER_CYCLE}" in text
    assert "unit_energy: 0.0" in text
    assert export.DRAM_BITS_PER_CYCLE == 128
    assert export.SCRATCHPAD_BITS_PER_CYCLE == 512


def test_the_timeloop_export_cuts_at_the_tile_boundary() -> None:
    """The two Timeloop levels are the tile loops and the loops inside a tile."""
    problem, choice = conv_pair()
    mapping = export.conv_mapping(problem, choice)
    text = export.timeloop_text(problem, mapping)
    dram = next(line for line in text.splitlines() if line.startswith("    factors:"))
    assert "P=2" in dram
    assert "target: PE" in text
    assert "K=8" in text.split("target: PE")[1]


def test_the_workload_form_is_the_installed_parsers_own() -> None:
    """The equation and the loop dimension names are copied, not remembered.

    Read out of the installed ZigZag rather than out of this file's memory of
    it, which is the same rule `test_scalesim_export.py` applies to the topology
    column order and for the same reason.
    """
    zigzag = pytest.importorskip("zigzag")
    source = (
        Path(zigzag.__file__).parent / "parser" / "onnx" / "conv_parser.py"
    ).read_text(encoding="utf-8")
    problem, _ = conv_pair()
    entry = export.workload_entry(problem)
    assert entry["equation"] in source
    spelled = '["' + '", "'.join(entry["loop_dims"]) + '"]'
    assert spelled in source

    gemm = (
        Path(zigzag.__file__).parent / "parser" / "onnx" / "gemm_parser.py"
    ).read_text(encoding="utf-8")
    matmul, _ = matmul_pair()
    assert '["C", "D", "K"]' in gemm
    assert export.workload_entry(matmul)["loop_dims"] == ["C", "D", "K"]


def test_the_spilling_configurations_carry_no_mapping_to_export() -> None:
    """Arm one ablates the tiling pass, so there is nothing to sweep there."""
    assert "spill-longest-range" not in export.TILING_CONFIGURATIONS
    assert "spill-cost" not in export.TILING_CONFIGURATIONS
    assert "tile-fused-recompute" in export.TILING_CONFIGURATIONS


def test_the_frozen_budgets_are_the_registrys_own() -> None:
    from npu_frontend.model_generator import MODELS

    assert export.TIGHT == {name: spec.tight_budget for name, spec in MODELS.items()}


def test_zigzag_evaluates_the_ordering_it_is_handed(tmp_path: Path) -> None:
    """The end to end property, read back off ZigZag's own evaluation.

    Everything else here checks that the export is well formed. This checks the
    only thing that makes the comparison mean anything: that the loop nest
    ZigZag scored is the loop nest the compiler chose.
    """
    pytest.importorskip("zigzag")
    problem, choice = conv_pair()
    mapping = export.conv_mapping(problem, choice)
    export.check_complete(problem, mapping)
    answer = export.run_zigzag(problem, mapping, 6464, tmp_path)
    requested = [f"{name}, {factor}" for name, factor in mapping.temporal if factor > 1]
    assert answer["evaluated"] == requested
    assert answer["latency"] > 0


def test_the_npu_stage_is_the_same_pipeline_at_the_same_budget(
    tmp_path: Path,
) -> None:
    """D-0059: `--emit npu` dropped the budget and nothing noticed.

    The tensor level half is described as the same pipeline stopped earlier, so a
    budget that reaches one stage and not the other makes that description false
    for every pass that consumes it. This is also the parser against the real
    compiler rather than against a copied snippet.
    """
    onnx = generate_model("resnet_block", tmp_path, batch=1)
    tight = export.npu_stage(onnx, 6464, None)
    loose = export.npu_stage(onnx, None, None)
    choices = export._choices(tight)
    assert choices, "resnet_block tiles at its frozen budget"
    shapes = export.problems(loose)
    for name in choices:
        assert name in shapes, f"{name} has no untiled shape to export against"
        assert shapes[name].sizes["OY"] == 8
