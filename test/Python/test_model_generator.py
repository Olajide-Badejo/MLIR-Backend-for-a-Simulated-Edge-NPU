# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The seeded model suite: structure, determinism, and that it all imports.

Section 15 asks for three things per model and this file is those three. A
structural test asserting the exported ONNX contains the operations it is
supposed to, a determinism test asserting two exports at the same seed produce
identical first layer weights, and `INPUT_SHAPES` asserted equal to the model
registry so the two cannot drift.

The structural tests are the ones that earn their keep over time. Their whole
job is to fail when a toolchain upgrade changes what an exporter emits, which is
how this project already lost its batch norm model once: the exporter folded the
batch norms away, `-npu-fold-batchnorm` had nothing to fold, and the flagship
pass went quietly dead. A structural assertion turns that into a red test.

The whole suite is generated once per session, into a temporary directory. No
`.onnx` file is ever committed; they are regenerated from the seed, which is why
`*.onnx` is in `.gitignore`.
"""

from __future__ import annotations

import re

import numpy as np
import onnx
import pytest
from npu_frontend import (
    GENERATOR_VERSION,
    INPUT_SHAPES,
    MODELS,
    generate_model,
    import_model_file,
)
from npu_frontend.model_generator import (
    DEFAULT_SEED,
    first_weight,
    node_counts,
    with_batch,
)


@pytest.fixture(scope="session")
def suite(tmp_path_factory: pytest.TempPathFactory) -> dict[str, onnx.ModelProto]:
    """Every model, generated once. Exporting is the expensive part here."""
    directory = tmp_path_factory.mktemp("model-suite")
    return {name: onnx.load(str(generate_model(name, directory))) for name in MODELS}


@pytest.fixture(scope="session")
def suite_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    directory = tmp_path_factory.mktemp("model-suite-paths")
    return {name: str(generate_model(name, directory)) for name in MODELS}


# =============================================================================
# The registry itself.
# =============================================================================


def test_the_suite_has_at_least_seven_models() -> None:
    """Section 0.3's scope: at least seven structurally distinct models."""
    assert len(MODELS) >= 7


def test_input_shapes_agrees_with_the_model_registry() -> None:
    """Section 15 asks that the two registries cannot drift.

    Deriving one from the other is the strongest form of that assertion, and
    this test is what stops somebody from later writing the second list out by
    hand as a convenience.
    """
    assert INPUT_SHAPES == {name: spec.input_shape for name, spec in MODELS.items()}


def test_every_model_declares_exactly_one_build_route() -> None:
    """Exported from PyTorch or built with the ONNX API, never both or neither."""
    for name, spec in MODELS.items():
        routes = [spec.torch_factory is not None, spec.onnx_builder is not None]
        assert sum(routes) == 1, name


def test_exactly_two_models_are_built_with_the_onnx_construction_api() -> None:
    """Section 15's exception covers two models and says why for each.

    Pinning the count means a third hand built model is a decision somebody has
    to make on purpose, with a reason, rather than a convenience that quietly
    removes exporter coverage from the suite.
    """
    hand_built = {
        name for name, spec in MODELS.items() if spec.onnx_builder is not None
    }
    assert hand_built == {"conv_bn_relu_stack", "dilated_stack"}


def test_export_options_are_not_shared_between_models() -> None:
    """Per model, never global.

    A global flag change would move LeNet's exported graph, which moves the
    golden tensors, which turns a model addition into a baseline regression.
    Distinct dictionary objects is what makes that structurally impossible
    rather than merely intended.
    """
    identities = [
        id(spec.export_options)
        for spec in MODELS.values()
        if spec.torch_factory is not None
    ]
    assert len(identities) == len(set(identities))


def test_no_model_reaches_the_legacy_exporter() -> None:
    """Section 15 forbids `dynamo=False`, and it is passed explicitly as True.

    The legacy exporter is deprecated and will be removed, and it is also the
    only way to get an unfolded batch norm out of PyTorch, so a future reader
    under time pressure has a strong incentive to reach for it. This test is the
    thing standing in the way.
    """
    for name, spec in MODELS.items():
        if spec.torch_factory is None:
            continue
        assert spec.export_options["dynamo"] is True, name


def test_the_generator_version_is_recorded() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", GENERATOR_VERSION)


# =============================================================================
# Structure, per model.
# =============================================================================


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_exported_graph_has_exactly_the_expected_nodes(
    name: str, suite: dict[str, onnx.ModelProto]
) -> None:
    """Exact counts, not a subset.

    A subset check passes on a graph that gained a node, and a gained node is
    how an exporter upgrade silently changes what the suite covers.
    """
    assert node_counts(suite[name]) == dict(MODELS[name].expected_nodes)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_is_at_the_pinned_opset(
    name: str, suite: dict[str, onnx.ModelProto]
) -> None:
    from npu_frontend import PINNED_OPSET

    imports = [(entry.domain, entry.version) for entry in suite[name].opset_import]
    assert imports == [("", PINNED_OPSET)]


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_declares_the_input_shape_the_registry_says(
    name: str, suite: dict[str, onnx.ModelProto]
) -> None:
    graph_input = suite[name].graph.input[0]
    declared = tuple(
        dimension.dim_value for dimension in graph_input.type.tensor_type.shape.dim
    )
    assert declared == MODELS[name].input_shape


def test_exactly_one_mul_survives_the_resnet_export(
    suite: dict[str, onnx.ModelProto],
) -> None:
    """Section 15 asks for this by name, and says what to do when it fails.

    `npu.mul` has no other model in the suite. If a future exporter optimizer
    folds the per channel scale into the preceding convolution's weights, this
    test fails loudly and the `Mul` moves into an ONNX API built model with the
    reason recorded, rather than the operator silently going unreachable.
    """
    assert node_counts(suite["resnet_block"]).get("Mul") == 1


def test_the_resnet_scale_is_a_channel_shaped_initializer(
    suite: dict[str, onnx.ModelProto],
) -> None:
    """And it is the carve out's shape, which is the reason the Mul is there."""
    model = suite["resnet_block"]
    mul = next(node for node in model.graph.node if node.op_type == "Mul")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    scale = next(name for name in mul.input if name in initializers)
    dims = tuple(initializers[scale].dims)
    assert dims in {(8,), (1, 8, 1, 1)}, dims


def test_exactly_one_transpose_closes_the_dilated_stack(
    suite: dict[str, onnx.ModelProto],
) -> None:
    model = suite["dilated_stack"]
    transposes = [node for node in model.graph.node if node.op_type == "Transpose"]
    assert len(transposes) == 1
    perm = next(a for a in transposes[0].attribute if a.name == "perm")
    assert list(perm.ints) == [0, 2, 3, 1]


def test_the_dilated_stack_carries_a_separate_channel_shaped_bias_add(
    suite: dict[str, onnx.ModelProto],
) -> None:
    """The shape `-npu-fuse-bias` exists for, held where the model is built.

    Three properties, and the pass reads each one. `conv1` has two inputs, so it
    is biasless and has somewhere to put a bias. An `Add` consumes its result and
    its second input is an initializer, so the addend is a constant. And that
    initializer is channel shaped rather than result shaped, which is what
    separates a bias from a residual.

    The `(1, C, 1, 1)` spelling is admitted alongside `(C,)` for the same reason
    `test_the_resnet_scale_is_a_channel_shaped_initializer` admits both: ONNX
    broadcasting aligns from the trailing axis, so the four dimensional form is
    the only one `onnx.checker` accepts here, and `docs/adr/0005` is the record
    of the importer normalising it to rank 1 on the way in.
    """
    model = suite["dilated_stack"]
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}

    conv1 = next(node for node in model.graph.node if node.name == "conv1")
    assert len(conv1.input) == 2, (
        "conv1 gained a Conv bias input, which leaves -npu-fuse-bias nothing to "
        "fuse into and reopens the suite gap P9 recorded"
    )

    add = next(node for node in model.graph.node if node.op_type == "Add")
    assert add.input[0] == conv1.output[0]
    addend = add.input[1]
    assert addend in initializers, "the addend is not a constant, so it is not a bias"
    dims = tuple(initializers[addend].dims)
    assert dims in {(5,), (1, 5, 1, 1)}, dims


def test_the_dilated_stack_really_dilates_and_really_pads_asymmetrically(
    suite: dict[str, onnx.ModelProto],
) -> None:
    """Both properties, because either alone would be a weaker model.

    A model with dilations and symmetric pads never exercises the pad ordering,
    and a model with asymmetric pads and no dilation never exercises the
    effective kernel term.
    """
    convolutions = [
        node for node in suite["dilated_stack"].graph.node if node.op_type == "Conv"
    ]
    dilations = [
        list(next(a for a in node.attribute if a.name == "dilations").ints)
        for node in convolutions
    ]
    pads = [
        list(next(a for a in node.attribute if a.name == "pads").ints)
        for node in convolutions
    ]
    assert all(any(d > 1 for d in entry) for entry in dilations)
    assert any(entry[:2] != entry[2:] for entry in pads)


def test_the_batch_norm_stack_still_holds_its_batch_norms(
    suite: dict[str, onnx.ModelProto],
) -> None:
    """The regression test for the defect this model exists to close.

    On a current PyTorch the exporter folds a batch norm into the preceding
    convolution and there is no documented way to stop it, so this model is
    built with the ONNX construction API and the nodes are here by construction.
    If a future toolchain change folds them anyway, this fails instead of
    `-npu-fold-batchnorm` silently going dead.
    """
    assert node_counts(suite["conv_bn_relu_stack"])["BatchNormalization"] == 2


def test_the_depthwise_block_reaches_the_group_equals_channels_path(
    suite: dict[str, onnx.ModelProto],
) -> None:
    convolutions = [
        node
        for node in suite["depthwise_separable"].graph.node
        if node.op_type == "Conv"
    ]
    groups = [
        next((a.i for a in node.attribute if a.name == "group"), 1)
        for node in convolutions
    ]
    assert max(groups) == 8


def test_the_batched_lenet_carries_the_batch_through_to_its_output(
    suite: dict[str, onnx.ModelProto],
) -> None:
    output = suite["lenet_batched"].graph.output[0]
    declared = tuple(d.dim_value for d in output.type.tensor_type.shape.dim)
    assert declared == (4, 10)


def test_the_suite_covers_every_converter_the_importer_registers(
    suite: dict[str, onnx.ModelProto],
) -> None:
    """Section 17.1: an importer operator needs a suite model that uses it.

    Not a nice to have. A converter with only an isolated unit test has never
    seen the shape an exporter actually produces, and the two differ often
    enough that the isolated test alone is not evidence.
    """
    from npu_frontend import CONVERTERS

    covered: set[str] = set()
    for model in suite.values():
        covered.update(node.op_type for node in model.graph.node)
    assert set(CONVERTERS) - covered == set()


# =============================================================================
# Determinism.
# =============================================================================


@pytest.mark.parametrize("name", sorted(MODELS))
def test_two_exports_at_the_same_seed_produce_identical_first_layer_weights(
    name: str, tmp_path
) -> None:
    """Section 15's determinism requirement, asserted bit exactly.

    The first layer's filter rather than a hash of the file, because a file hash
    also covers the producer version and the node names, and a change to either
    of those is not a change to what the model computes.
    """
    first = onnx.load(str(generate_model(name, tmp_path / "a", seed=DEFAULT_SEED)))
    second = onnx.load(str(generate_model(name, tmp_path / "b", seed=DEFAULT_SEED)))
    np.testing.assert_array_equal(first_weight(first), first_weight(second))


def test_a_different_seed_produces_different_weights(tmp_path) -> None:
    """The other half of the determinism claim.

    Without it a generator that ignored its seed entirely would pass the test
    above, which would make the seed decorative.
    """
    first = onnx.load(str(generate_model("lenet", tmp_path / "a", seed=1)))
    second = onnx.load(str(generate_model("lenet", tmp_path / "b", seed=2)))
    assert not np.array_equal(first_weight(first), first_weight(second))


def test_the_hand_built_models_are_deterministic_too(tmp_path) -> None:
    """They draw from numpy rather than torch, so they are seeded separately."""
    for name in ("conv_bn_relu_stack", "dilated_stack"):
        first = onnx.load(str(generate_model(name, tmp_path / "a", seed=99)))
        second = onnx.load(str(generate_model(name, tmp_path / "b", seed=99)))
        assert first.SerializeToString() == second.SerializeToString(), name


# =============================================================================
# Every model imports.
# =============================================================================


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_imports_to_verified_npu_ir(
    name: str, suite_paths: dict[str, str]
) -> None:
    """The end of the P3 gate: seven models, seven verified modules.

    `import_model_file` returns the text `npu-opt` printed, so reaching this
    assertion at all means the module parsed, every verifier ran and passed, and
    no operation carried a discardable attribute.
    """
    ir = import_model_file(suite_paths[name])
    assert ir.startswith("#loc") or ir.lstrip().startswith("module")
    assert "func.func @main" in ir


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_compute_operation_in_every_model_has_its_own_destination(
    name: str, suite_paths: dict[str, str]
) -> None:
    """The gate item, over the whole suite rather than over one fixture.

    Two operations sharing one `tensor.empty` would typecheck and would alias
    two live values, which the allocator would later see as one buffer written
    twice. Comparing the sorted lists catches both a missing destination and a
    reused one.
    """
    ir = import_model_file(suite_paths[name])
    outs = re.findall(r"outs\((%\w+) :", ir)
    empties = re.findall(r"(%\w+) = tensor\.empty\(\)", ir)
    assert sorted(outs) == sorted(empties)
    assert len(set(outs)) == len(outs)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_keeps_its_batch_dimension_end_to_end(
    name: str, suite_paths: dict[str, str]
) -> None:
    """The function's argument and result agree with the registry's batch size.

    The batched LeNet is what this is really about, but running it over every
    model costs nothing and means a model added later cannot quietly lose its
    batch.
    """
    batch = MODELS[name].input_shape[0]
    ir = import_model_file(suite_paths[name])
    signature = re.search(r"func\.func @main\((.*?)\) -> (\S+)", ir, re.DOTALL)
    assert signature is not None
    assert f"tensor<{batch}x" in signature.group(1)
    assert signature.group(2).startswith(f"tensor<{batch}x")


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_imports_at_a_second_seed(name: str, tmp_path) -> None:
    """Structure is a property of the model, not of the draw.

    Marked slow because it regenerates the whole suite a second time, which is
    the expensive half of this file, and because what it proves is a second
    order property: the fast tests already prove every model imports.
    """
    ir = import_model_file(str(generate_model(name, tmp_path, seed=DEFAULT_SEED + 1)))
    assert "func.func @main" in ir


# =============================================================================
# The batch override, added at P8 for Section 17.4's matrix.
# =============================================================================


@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_batch_override_changes_the_batch_and_nothing_else(
    name: str, tmp_path
) -> None:
    """Section 17.4 sweeps the batch and Section 15 pins one per model.

    The two are reconciled by making the batch a parameter of an export rather
    than a second registry, and the property that makes that legitimate is
    asserted here: the weights do not move. If they did, a batch 4 cell and a
    batch 1 cell would be two different models and the matrix would be
    comparing one to the other.
    """
    at_registry = onnx.load(str(generate_model(name, tmp_path)))
    at_seven = onnx.load(str(generate_model(name, tmp_path, batch=7)))

    assert at_seven.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 7
    assert at_seven.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 7
    assert node_counts(at_seven) == node_counts(at_registry)
    np.testing.assert_array_equal(first_weight(at_seven), first_weight(at_registry))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_registrys_own_batch_writes_the_registrys_own_file(
    name: str, tmp_path
) -> None:
    """A caller sweeping [1, 4] exports eleven files, not fourteen.

    Naming the file after the effective batch only when it differs from the
    registry's is what lets the matrix ask for `lenet` at 1 and `lenet_batched`
    at 4 and get the files the rest of the project already knows by name.
    """
    registry_batch = MODELS[name].input_shape[0]
    default = generate_model(name, tmp_path)
    explicit = generate_model(name, tmp_path, batch=registry_batch)
    assert default == explicit
    assert default.name == f"{name}.onnx"

    other = generate_model(name, tmp_path, batch=registry_batch + 1)
    assert other.name == f"{name}-n{registry_batch + 1}.onnx"


def test_a_batch_that_is_not_a_batch_is_refused() -> None:
    for batch in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            with_batch((1, 3, 8, 8), batch)
