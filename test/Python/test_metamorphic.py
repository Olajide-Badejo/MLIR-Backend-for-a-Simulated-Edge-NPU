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
equality, not a tolerance. A tolerance here would be a place for a real
disagreement to hide.

**The tolerance question P8 left on P9's desk, measured and answered.** P8
recorded that fusion changes accumulation order, so these relations would stop
being exactly equal above `-O0`, and said the answer was P9's to record rather
than to inherit. It is one cell, and the mechanism is not the one P8 expected.

Measured on 2026-08-31 over the whole cross product of four relations, seven
models and three levels, which is eighty four cells of which twenty nine apply:

- at `-O0` and `-O1`, **every applicable cell agrees to exactly zero**;
- at `-O2`, every cell agrees to exactly zero **except**
  `convolution_split` on `conv_bn_relu_stack`, which moves by **2.98e-08**
  absolute, 1.30e-07 against the answer's largest magnitude.

**The mechanism, named rather than attributed to fusion in general.** A relation
compares two compilations *at the same level*, so a pass that reassociates does
so on both sides and cancels. What does not cancel is a rewrite that changes
which passes can **match**. `convolution_split` replaces one convolution with
two over channel groups and a concatenation, so in the variant the batch norm's
producer is an `npu.concat` rather than an `npu.conv2d` and
`-npu-fold-batchnorm` declines. The original folds and the variant does not, and
the 2.98e-08 is the fold's own movement showing up on one side of a comparison
of two programs that are still computing the same function.

It is therefore not a defect and not a reason to weaken the relation. The bound
is 1e-6, which is the band Section 17.6 sets for this phase's numerics and the
same band `docs/BREAKING_CHANGES.md` declares the movement in, and it applies
**only at a level whose pipeline runs a pass that can be made unfirable by a
rewrite**. Which levels those are is read out of the compiler through
`REASSOCIATING_PASSES` below, so `-O0` and `-O1` keep byte equality and a level
that stopped folding would get it back without an edit here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import pytest
from npu_frontend import MODELS, compile_model, generate_model, run_program
from npu_frontend.compile import (
    describe_pipeline,
    level_description,
    levels_that_eliminate_dead_code,
)
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

#: Every level the compiler builds. *Added at P9.* A relation compiles the
#: original and the variant **at the same level** and compares them, so the
#: level is a dimension of the oracle rather than a setting: a rewrite that is
#: semantics preserving at `-O0` and not at `-O2` is exactly the pass bug this
#: file exists to find.
LEVELS = (0, 1, 2)

#: `-O0` still, for the claims that are about the importer rather than about a
#: level.
LEVEL = 0

#: The model that exists to have parallel branches, and therefore the only one
#: `node_order_permutation` can act on. Named here so that a suite change that
#: removed the branching would fail rather than quietly leave the relation with
#: nowhere to apply.
BRANCHING_MODEL = "inception_block"

#: The passes a semantics preserving rewrite of the input model can stop from
#: firing, so that the original and the variant are compiled by different
#: arithmetic even though they compute the same function.
#:
#: One entry at P9. `-npu-fold-batchnorm` matches a batch norm whose producer is
#: a convolution, and `convolution_split` turns that producer into a
#: concatenation, so the original folds and the variant does not.
#: `-npu-fuse-bias` and `-npu-fuse-ops` are not here and belong here if they
#: ever start moving numbers: both are measured to be bit exact, the first
#: because the kernel adds the bias to the same `f32` accumulator either way and
#: the second because the lowering flattens the region into the same
#: instructions.
#:
#: `test_the_reassociating_passes_are_passes_this_compiler_has` asserts every
#: name here is a pass at `-O2`, so a rename is a red test rather than a silently
#: empty set that would restore byte equality and hide a real movement.
REASSOCIATING_PASSES: Final[frozenset[str]] = frozenset({"npu-fold-batchnorm"})

#: The band a relation may move within at a level that runs one of those.
#:
#: 1e-6 is Section 17.6's class for this phase, and it is the same band
#: `docs/BREAKING_CHANGES.md` declares the level to level movement in, so the
#: two numbers in this project's documentation are one number. Observed on
#: 2026-08-31: 2.98e-08 absolute and 1.30e-07 against the answer's largest
#: magnitude, in one cell of eighty four. Never widened to make a cell pass; a
#: cell that needs more is a finding, and the finding is written down.
METAMORPHIC_TOLERANCE: Final[float] = 1e-6


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    directory = tmp_path_factory.mktemp("metamorphic")
    yield {name: generate_model(name, directory) for name in MODELS}


def run(
    model: onnx.ModelProto,
    name: str,
    input_class: str = "normal",
    *,
    level: int = LEVEL,
):
    """Compiles and runs one model, and returns the answer and the count."""
    program = compile_model(model, level=level, emit="nbin")
    assert program.binary is not None
    arrays = make_inputs(input_class, program.input_shapes, model=name, batch=1)
    return arrays, run_program(program.binary, arrays, program.output_shapes)


def run_with(model: onnx.ModelProto, arrays: list[np.ndarray], *, level: int = LEVEL):
    program = compile_model(model, level=level, emit="nbin")
    assert program.binary is not None
    return run_program(program.binary, arrays, program.output_shapes)


def relations_are_exact_at(level: int) -> bool:
    """Whether a relation must produce a bit identical answer at this level.

    Read out of the compiler's own pipeline description rather than from a list
    of level numbers, so a pass moved between levels changes this answer without
    an edit here. Exact means exact: the assertion is `assert_array_equal` and
    not a tolerance of zero, because the two say the same thing and only one of
    them says it in the failure message.
    """
    running = {entry["pass"] for entry in level_description(level)["passes"]}
    return not (running & REASSOCIATING_PASSES)


def assert_relation_agreed(
    produced: np.ndarray, expected: np.ndarray, cell: str, level: int
) -> None:
    """The comparison, exact where it can be and bounded where it cannot."""
    if relations_are_exact_at(level):
        np.testing.assert_array_equal(
            produced,
            expected,
            err_msg=(
                f"{cell} moved the answer at a level that runs no pass a "
                f"rewrite can stop from firing, so the two programs were "
                f"compiled by the same arithmetic and a difference of one bit "
                f"is a compiler defect rather than a tolerance to widen."
            ),
        )
        return

    difference = np.abs(produced.astype(np.float64) - expected.astype(np.float64))
    absolute = float(difference.max())
    assert absolute <= METAMORPHIC_TOLERANCE, (
        f"{cell}: the absolute difference is {absolute:.3e}, above the band of "
        f"{METAMORPHIC_TOLERANCE:.3e}. This band is Section 17.6's class for "
        f"this phase and is not to be widened to make a cell pass; record the "
        f"measured value and name the pass whose matching the rewrite changed."
    )

    scale = float(np.abs(expected.astype(np.float64)).max())
    if scale == 0.0:
        # Nothing to be relative to, so the stronger claim is asserted instead.
        assert np.array_equal(produced, np.zeros_like(produced)), (
            f"{cell}: the original is exactly zero everywhere and the variant "
            f"is not"
        )
        return

    relative = absolute / scale
    assert relative <= METAMORPHIC_TOLERANCE, (
        f"{cell}: the difference relative to the answer's largest magnitude "
        f"{scale:.3e} is {relative:.3e}, above {METAMORPHIC_TOLERANCE:.3e}"
    )


# ---------------------------------------------------------------------------
# The relations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize(
    "relation_name",
    [entry.name for entry in RELATIONS],
    ids=[e.name for e in RELATIONS],
)
def test_a_relation_preserves_the_answer(
    models: dict[str, Path], model_name: str, relation_name: str, level: int
) -> None:
    """The gate item: the metamorphic relations pass on every model.

    A relation that cannot act on a model is reported as inapplicable rather
    than as a pass, and `test_every_relation_reaches_at_least_one_model` is what
    keeps an inapplicable relation from becoming a relation that applies
    nowhere.

    **The level axis arrived at P9, and with it the tolerance question P8 left
    on its desk.** The module docstring carries the measurement and the
    mechanism; the short version is that a relation compares two compilations at
    the same level, so a reassociating pass cancels unless the rewrite changed
    what it could **match**, and exactly one of the four does. The comparison is
    exact wherever no such pass runs, which is read out of the compiler rather
    than written down as a list of level numbers.
    """
    entry = relation(relation_name)
    original = onnx.load(str(models[model_name]))
    try:
        variant = entry.rewrite(original)
    except NotApplicable as reason:
        pytest.skip(f"{relation_name} does not apply to {model_name}: {reason}")

    arrays, expected = run(original, model_name, level=level)
    produced = run_with(variant, arrays, level=level)

    assert len(produced.outputs) == len(expected.outputs)
    for got, want in zip(produced.outputs, expected.outputs, strict=True):
        assert_relation_agreed(
            got, want, f"{relation_name} on {model_name} at -O{level}", level
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


def test_the_reassociating_passes_are_passes_this_compiler_has() -> None:
    """A name in `REASSOCIATING_PASSES` that is not a pass would be a set that
    never intersects anything, and a set that never intersects anything restores
    byte equality everywhere and hides the one cell that really does move.

    So each name is checked against the compiler's own `-O2` list, and the
    levels that are exact are asserted by value beside it, because reading the
    rule out of the compiler is only half of the protection: the other half is
    saying what the answer is expected to be today.
    """
    at_o2 = {entry["pass"] for entry in level_description(2)["passes"]}
    for name in REASSOCIATING_PASSES:
        assert name in at_o2, (
            f"{name} is in REASSOCIATING_PASSES and is not a pass at -O2, so "
            f"the set intersects nothing and every level is treated as exact"
        )

    assert relations_are_exact_at(0) is True
    assert relations_are_exact_at(1) is True
    assert relations_are_exact_at(2) is False


def test_the_one_relation_that_moves_moves_by_what_was_recorded(
    models: dict[str, Path],
) -> None:
    """The measurement behind the band, asserted rather than left in a comment.

    Section 17.4's rule is that a tolerance is set close to the observed value
    and the observed value is recorded. This is the observation, run: the single
    cell that moves is `convolution_split` on `conv_bn_relu_stack` at `-O2`, and
    it moves by about 3e-08, which is thirty times inside the 1e-6 band. A cell
    that had crept to 9e-07 would still pass the band and would fail here, which
    is the point of measuring rather than only bounding.
    """
    original = onnx.load(str(models["conv_bn_relu_stack"]))
    variant = relation("convolution_split").rewrite(original)
    arrays, expected = run(original, "conv_bn_relu_stack", level=2)
    produced = run_with(variant, arrays, level=2)

    worst = max(
        float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        for got, want in zip(produced.outputs, expected.outputs, strict=True)
    )
    assert 0.0 < worst < 1e-07, (
        f"the recorded movement was 2.98e-08 and this run measured {worst:.3e}. "
        f"A movement of exactly zero would mean the batch norm fold stopped "
        f"firing on the original; a larger one is a finding to record."
    )


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


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_a_dead_subgraph_changes_no_output(
    models: dict[str, Path], model_name: str, level: int
) -> None:
    """The stronger half of the gate item, at every level.

    Bit identical, with no tolerance, because nothing the live graph computes
    depends on any of the injected work. It also stresses the allocator with
    interval sets no hand written test produces, which Section 17.3a names as
    the second reason to do this at all.

    At `-O1` and `-O2` the injected work is also *removed*, and that is the
    other half asserted separately below. Here the claim is only that it made no
    difference, which is the claim that holds whether or not a pass removed it.
    """
    original = onnx.load(str(models[model_name]))
    injected = inject_dead_subgraph(original)

    for input_class in INPUT_CLASSES:
        arrays, expected = run(original, model_name, input_class, level=level)
        produced = run_with(injected, arrays, level=level)
        for got, want in zip(produced.outputs, expected.outputs, strict=True):
            np.testing.assert_array_equal(
                got, want, err_msg=f"{model_name} at -O{level}"
            )


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_a_dead_subgraph_adds_exactly_what_it_brought_at_minus_o_zero(
    models: dict[str, Path], model_name: str
) -> None:
    """`-O0` eliminates nothing, so the count moves by exactly three.

    The gate asks that a dead subgraph leave the `-O2` instruction count
    unchanged, and since P9 the test below asserts exactly that at both levels
    that can satisfy it. This one is the opposite claim at the level that
    cannot, and it is kept rather than retired because it is just as falsifiable
    and it is the control: without it, a `-O2` count that did not move would be
    consistent with an injection that brought nothing.

    A count that grew by four would mean the injection cost something it did not
    declare; one that grew by two would mean a pass nobody registered removed
    something.
    """
    original = onnx.load(str(models[model_name]))
    injected = inject_dead_subgraph(original)

    arrays, before = run(original, model_name)
    after = run_with(injected, arrays)
    assert after.instructions == before.instructions + DEAD_NODE_COUNT


def test_the_levels_that_eliminate_dead_code_are_the_ones_with_a_pass_that_does() -> (
    None
):
    """Which is why the gate's `-O2` clause is answered the way it is.

    Read from the compiler's own pipeline description, not from a list here.
    Until P9 this test asserted the set was **empty** and said that the phase
    which made the claim true would be the phase that deleted the assertion.
    This is that deletion, and what replaced it is the same claim with the
    answer filled in: `-O1` and `-O2` both run `-canonicalize`, which removes
    every `npu` operation nothing reads because they all carry `Pure`, and `-O2`
    also runs `-symbol-dce`.

    The set is still read out of the compiler, so a level that stopped running
    such a pass would drop out of the parametrization below and this assertion
    is what makes that a red test rather than a quieter one.
    """
    assert levels_that_eliminate_dead_code() == [1, 2]

    marked = {
        int(row["level"]): [
            entry["pass"] for entry in row["passes"] if entry["eliminates_dead_code"]
        ]
        for row in describe_pipeline()["levels"]
    }
    assert marked[0] == []
    assert marked[1] == ["canonicalize"]
    assert marked[2] == ["canonicalize", "canonicalize", "symbol-dce"]


@pytest.mark.parametrize("level", levels_that_eliminate_dead_code())
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_a_dead_subgraph_does_not_change_the_instruction_count(
    models: dict[str, Path], model_name: str, level: int
) -> None:
    """The gate's own wording, at every level that can satisfy it.

    The parametrization reads the level set from the compiler, so P9 filled it
    by marking two passes rather than by editing this file. It was empty at P8
    and the test above asserted the emptiness, so the vacuous parametrization
    was declared rather than left to be noticed.
    """
    original = onnx.load(str(models[model_name]))
    injected = inject_dead_subgraph(original)
    arrays, before = run(original, model_name, level=level)
    after = run_with(injected, arrays, level=level)
    assert after.instructions == before.instructions, (
        f"{model_name} at -O{level}: the injected subgraph feeds nothing and "
        f"this level runs a pass marked as eliminating dead code, so the count "
        f"must not move. It went {before.instructions} -> {after.instructions}."
    )
