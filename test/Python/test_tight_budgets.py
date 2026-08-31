# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The per model tight budgets of Section 15, measured at P8 and frozen.

`docs/adr/0008-per-model-tight-scratchpad-budgets.md` is the record: what was
measured, what the two deviations from Section 15's rule are, and why the fixed
fraction that section asks for is inoperative until tiling lands at P13.

**What these tests assert and what they deliberately do not.** They assert that
each frozen constant still allocates, still produces the same answer as the
default budget, and still spills where the record says it spills. They do *not*
assert that each constant is still the smallest budget that allocates. That
floor moving is exactly what an allocator improvement looks like, and a test
that failed on one would be a test pushing somebody to re-measure the constants,
which Section 15 calls a prime directive breach dressed up as a re-measurement.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from npu_frontend import (
    MODELS,
    CompileError,
    compile_model,
    generate_model,
    run_program,
)
from npu_frontend.input_classes import INPUT_CLASSES, make_inputs
from npu_frontend.model_generator import DEFAULT_BUDGET, TIGHT_BUDGETS

#: What the record says each model does at its tight budget: the spilled buffer
#: count and the DMA operations that carries. Written here from the record
#: rather than measured, so a change in either direction is a red test that
#: sends the reader to the record.
EXPECTED_SPILLS: dict[str, tuple[int, int]] = {
    "lenet": (0, 0),
    "depthwise_separable": (0, 0),
    "resnet_block": (1, 3),
    "inception_block": (3, 8),
    "conv_bn_relu_stack": (0, 0),
    "dilated_stack": (0, 0),
    "lenet_batched": (0, 0),
}


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    directory = tmp_path_factory.mktemp("tight-budgets")
    yield {name: generate_model(name, directory) for name in MODELS}


def attribute(text: str, name: str) -> int:
    import re

    match = re.search(rf"npuisa\.{name} = (\d+)", text)
    assert match is not None, f"the lowered IR carries no npuisa.{name}"
    return int(match.group(1))


def test_the_registry_carries_a_tight_budget_for_every_model() -> None:
    """A model added without one is an error at import, not a default.

    `tight_budget` has no default on `ModelSpec`, so this is a property of the
    dataclass rather than of this test; the test is here to say out loud that
    the property is intended.
    """
    assert set(TIGHT_BUDGETS) == set(MODELS)
    assert all(value > 0 for value in TIGHT_BUDGETS.values())


def test_the_anchor_models_tight_budget_is_the_recorded_one() -> None:
    """Section 15 pins it so LeNet's numbers stay comparable over time.

    The literal is here as well as in the registry on purpose. Section 15 says a
    moved tight budget moves every tight budget cell in the project's history at
    once, so the anchor's constant gets an assertion that names the number, and
    changing it means changing two files and writing a BREAKING_CHANGES entry.
    """
    assert TIGHT_BUDGETS["lenet"] == 194625


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_tight_budget_still_allocates(models: dict[str, Path], name: str) -> None:
    """The property the project depends on, checked rather than assumed.

    Not that it is still the floor. An allocator that improved would move the
    floor, and a test that failed on that would be pushing somebody to
    re-measure a frozen constant.
    """
    lowered = compile_model(
        models[name], level=0, emit="npuisa", budget=TIGHT_BUDGETS[name]
    )
    assert lowered.text is not None
    assert attribute(lowered.text, "scratchpad_budget") == TIGHT_BUDGETS[name]
    assert attribute(lowered.text, "scratchpad_bytes") <= TIGHT_BUDGETS[name]


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_tight_budget_is_at_most_the_default(
    models: dict[str, Path], name: str
) -> None:
    """A tight budget above the default would not be tight."""
    assert TIGHT_BUDGETS[name] <= DEFAULT_BUDGET


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_tight_budget_spills_what_the_record_says(
    models: dict[str, Path], name: str
) -> None:
    """Two models spill and five do not, and the record says which.

    Naming both halves is the honest form. Five tight budget cells measure that
    the allocator is stable under a budget change and nothing more, and a test
    that only checked the two spilling models would let the other five drift
    into spilling or out of it unnoticed.
    """
    lowered = compile_model(
        models[name], level=0, emit="npuisa", budget=TIGHT_BUDGETS[name]
    )
    assert lowered.text is not None
    spills = attribute(lowered.text, "spill_count")
    dma = attribute(lowered.text, "spill_dma_count")
    assert (spills, dma) == EXPECTED_SPILLS[name], (
        f"{name} spills ({spills}, {dma}) where "
        f"docs/adr/0008-per-model-tight-scratchpad-budgets.md records "
        f"{EXPECTED_SPILLS[name]}. If the allocator changed on purpose, the "
        f"record changes with it."
    )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_tight_budget_does_not_move_the_answer(
    models: dict[str, Path], name: str
) -> None:
    """A spill is a DMA round trip, so the arithmetic is untouched.

    Bit identical, over every input class. This is what makes the golden tensors
    of Section 17.6 per level rather than per level and budget: the budget
    changes the instruction stream and not the answer.
    """
    batch = MODELS[name].input_shape[0]
    generous = compile_model(models[name], level=0, emit="nbin")
    tight = compile_model(
        models[name], level=0, emit="nbin", budget=TIGHT_BUDGETS[name]
    )
    assert generous.binary is not None and tight.binary is not None

    for input_class in INPUT_CLASSES:
        arrays = make_inputs(
            input_class, generous.input_shapes, model=name, batch=batch
        )
        expected = run_program(generous.binary, arrays, generous.output_shapes)
        produced = run_program(tight.binary, arrays, tight.output_shapes)
        for got, want in zip(produced.outputs, expected.outputs, strict=True):
            np.testing.assert_array_equal(got, want, err_msg=f"{name}-{input_class}")


@pytest.mark.parametrize("name", sorted(MODELS))
def test_a_budget_below_the_tight_one_is_refused_with_both_numbers(
    models: dict[str, Path], name: str
) -> None:
    """The floor is a floor, and the refusal names the numbers.

    A step of one allocator alignment below the recorded constant. The record
    says these are the smallest allocatable budgets at 64 byte granularity on
    the day they were measured; if the allocator later places one of them, this
    test fails and the record gets a note rather than the constant getting a
    new value.
    """
    with pytest.raises(CompileError) as failure:
        compile_model(
            models[name], level=0, emit="npuisa", budget=TIGHT_BUDGETS[name] - 64
        )
    message = str(failure.value)
    assert str(TIGHT_BUDGETS[name] - 64) in message
    assert "too small" in message
