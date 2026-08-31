# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 17.3a's metamorphic relations and its dead subgraph injection.

A metamorphic relation rewrites the *input model*, compiles the original and the
variant at the same level, and asserts they agree. It needs no external runtime,
so it reaches operator combinations no model in the suite contains, and it is
the oracle that found over 400 erroneous compilation inputs across four
industrial deep learning compilers.

**Four of the five relations are here and the fifth cannot be written.** Pad
then slice back needs an ONNX `Pad`, which this importer refuses by name for the
reason Section 11 gives, and an ONNX `Slice`, which has no converter. Adding
either so that a test could use it would be growing the operator set to satisfy
a test, which is what law 2 exists to prevent. It is recorded in
`metamorphic.NOT_IMPLEMENTED` with that reason, and a test below asserts the
recorded reason is still the true one.

**Agreement here is exact, and that is worth stating rather than assuming.**
None of the four relations changes the order any output element's terms are
summed in: an identity is erased at import, a transpose and its inverse move
elements without arithmetic, splitting a convolution over output channels leaves
each output element's own reduction untouched, and permuting independent nodes
changes nothing at all about what is computed. So the assertion is byte
equality, not a tolerance. Measured on 2026-08-31: every applicable cell agreed
to exactly zero. A tolerance here would be a place for a real disagreement to
hide.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import pytest
from npu_frontend import MODELS, compile_model, generate_model, run_program
from npu_frontend.compile import levels_that_eliminate_dead_code
from npu_frontend.input_classes import INPUT_CLASSES, make_inputs
from npu_frontend.metamorphic import (
    DEAD_NODE_COUNT,
    NOT_IMPLEMENTED,
    RELATIONS,
    NotApplicable,
    applicable_relations,
    inject_dead_subgraph,
    relation,
)

LEVEL = 0

#: The model that exists to have parallel branches, and therefore the only one
#: `node_order_permutation` can act on. Named here so that a suite change that
#: removed the branching would fail rather than quietly leave the relation with
#: nowhere to apply.
BRANCHING_MODEL = "inception_block"


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    directory = tmp_path_factory.mktemp("metamorphic")
    yield {name: generate_model(name, directory) for name in MODELS}


def run(model: onnx.ModelProto, name: str, input_class: str = "normal"):
    """Compiles and runs one model, and returns the answer and the count."""
    program = compile_model(model, level=LEVEL, emit="nbin")
    assert program.binary is not None
    arrays = make_inputs(input_class, program.input_shapes, model=name, batch=1)
    return arrays, run_program(program.binary, arrays, program.output_shapes)


def run_with(model: onnx.ModelProto, arrays: list[np.ndarray]):
    program = compile_model(model, level=LEVEL, emit="nbin")
    assert program.binary is not None
    return run_program(program.binary, arrays, program.output_shapes)


# ---------------------------------------------------------------------------
# The relations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize(
    "relation_name",
    [entry.name for entry in RELATIONS],
    ids=[e.name for e in RELATIONS],
)
def test_a_relation_preserves_the_answer(
    models: dict[str, Path], model_name: str, relation_name: str
) -> None:
    """The gate item: the metamorphic relations pass on every model.

    A relation that cannot act on a model is reported as inapplicable rather
    than as a pass, and `test_every_relation_reaches_at_least_one_model` is what
    keeps an inapplicable relation from becoming a relation that applies
    nowhere.
    """
    entry = relation(relation_name)
    original = onnx.load(str(models[model_name]))
    try:
        variant = entry.rewrite(original)
    except NotApplicable as reason:
        pytest.skip(f"{relation_name} does not apply to {model_name}: {reason}")

    arrays, expected = run(original, model_name)
    produced = run_with(variant, arrays)

    assert len(produced.outputs) == len(expected.outputs)
    for got, want in zip(produced.outputs, expected.outputs, strict=True):
        # Byte equality, not a tolerance. None of these rewrites changes the
        # order any output element's terms are summed in, so a difference of one
        # bit is a defect rather than a rounding artefact.
        np.testing.assert_array_equal(
            got,
            want,
            err_msg=(
                f"{relation_name} on {model_name} moved the answer. This "
                f"relation is semantics preserving by construction, so a "
                f"difference here is a compiler defect and not a tolerance to "
                f"widen."
            ),
        )


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_identity_insertion_is_erased_at_import(
    models: dict[str, Path], model_name: str
) -> None:
    """A stronger claim than agreement, asserted where it holds.

    `convert_identity` binds its output to its input rather than emitting an
    operation, so the variant's `npu` IR is byte identical to the original's.
    Agreement on the answer would follow from that trivially; asserting the IR
    is what says the importer erased the node rather than compiled it away
    later.
    """
    original = onnx.load(str(models[model_name]))
    variant = relation("identity_insertion").rewrite(original)
    assert len(variant.graph.node) > len(original.graph.node)

    before = compile_model(original, level=LEVEL, emit="npu")
    after = compile_model(variant, level=LEVEL, emit="npu")
    assert before.text == after.text


def test_every_relation_reaches_at_least_one_model(models: dict[str, Path]) -> None:
    """A relation that applies nowhere is a relation that tests nothing.

    The failure mode of an inapplicable relation is that every cell skips and
    the suite stays green. This is the check that turns that into a red test.
    """
    reached: dict[str, list[str]] = {entry.name: [] for entry in RELATIONS}
    for model_name, path in sorted(models.items()):
        for name in applicable_relations(onnx.load(str(path))):
            reached[name].append(model_name)

    empty = [name for name, names in reached.items() if not names]
    assert not empty, f"these relations apply to no model in the suite: {empty}"

    # And the one that is expected to reach exactly the branching model does.
    # A straight line graph has one topological order, so this relation applying
    # only here is the relation rather than a limitation, and naming the model
    # means a suite change that flattened it fails loudly.
    assert reached["node_order_permutation"] == [BRANCHING_MODEL]


def test_the_relation_that_cannot_be_written_still_cannot_be() -> None:
    """Section 17.3a's fifth relation, and the reason it is absent.

    Recorded rather than dropped, and checked rather than recorded once: the
    reason is that `Pad` is refused by name and `Slice` has no converter, so if
    either ever gains one this assertion fails and the relation gets written.
    """
    from npu_frontend import CONVERTERS, DEFERRED

    assert "pad_then_slice" in NOT_IMPLEMENTED
    assert "Pad" in DEFERRED
    assert "Pad" not in CONVERTERS
    assert "Slice" not in CONVERTERS
    assert "Slice" not in DEFERRED


# ---------------------------------------------------------------------------
# Dead subgraph injection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_a_dead_subgraph_changes_no_output(
    models: dict[str, Path], model_name: str
) -> None:
    """The half of the gate item that is assertable at this phase.

    Bit identical, with no tolerance, because nothing the live graph computes
    depends on any of the injected work. It also stresses the allocator with
    interval sets no hand written test produces, which Section 17.3a names as
    the second reason to do this at all.
    """
    original = onnx.load(str(models[model_name]))
    injected = inject_dead_subgraph(original)

    for input_class in INPUT_CLASSES:
        arrays, expected = run(original, model_name, input_class)
        produced = run_with(injected, arrays)
        for got, want in zip(produced.outputs, expected.outputs, strict=True):
            np.testing.assert_array_equal(got, want, err_msg=model_name)


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_a_dead_subgraph_adds_exactly_what_it_brought_at_minus_o_zero(
    models: dict[str, Path], model_name: str
) -> None:
    """`-O0` eliminates nothing, so the count moves by exactly three.

    The gate asks that a dead subgraph leave the `-O2` instruction count
    unchanged. `-O2` does not exist until P9 and `-O0`'s pipeline contains no
    pass that removes anything, so at this phase the honest form of the same
    check is the exact opposite claim, which is just as falsifiable: the count
    grows by exactly the instructions the injection brought, and by no more.

    A count that grew by four would mean the injection cost something it did not
    declare; one that grew by two would mean a pass nobody registered removed
    something.
    """
    original = onnx.load(str(models[model_name]))
    injected = inject_dead_subgraph(original)

    arrays, before = run(original, model_name)
    after = run_with(injected, arrays)
    assert after.instructions == before.instructions + DEAD_NODE_COUNT


def test_no_level_this_compiler_builds_eliminates_dead_code() -> None:
    """Which is why the gate's `-O2` clause is answered the way it is.

    Read from the compiler's own pipeline description, not from a list here.
    When P9 adds `-canonicalize` to `-O1` and `-O2` this test goes red, and it
    should: the phase that makes the claim true is the phase that deletes this
    assertion and lets the parametrized one below start doing work.
    """
    assert levels_that_eliminate_dead_code() == []


@pytest.mark.parametrize("level", levels_that_eliminate_dead_code())
def test_a_dead_subgraph_does_not_change_the_instruction_count(
    models: dict[str, Path], level: int
) -> None:
    """The gate's own wording, at every level that can satisfy it.

    Empty at P8, because no level this compiler builds eliminates dead code, and
    the test above asserts that emptiness rather than leaving it to be noticed.
    The parametrization reads the level set from the compiler, so P9 fills it by
    adding a pass and not by editing this file.
    """
    for model_name, path in models.items():
        original = onnx.load(str(path))
        injected = inject_dead_subgraph(original)
        arrays, before = run(original, model_name)
        after = run_with(injected, arrays)
        assert after.instructions == before.instructions, model_name
