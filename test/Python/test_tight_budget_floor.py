# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The floor search of `experiments/tight_budget_floor.py`, over predicates.

The search is separated from the compiler for the same reason
`ScratchpadAllocation.h` is separated from the pass: the interesting failure is
in the search rule and not in the invocation, and a test that had to compile a
model to reach the rule could not reach the shapes that matter. A predicate is
three lines here and a non monotone allocator is not.

**The shape that matters is the non monotone one, and it is not hypothetical.**
With `-npu-fuse-ops` left out, `conv_bn_relu_stack` fails to allocate at 4928
bytes on this tree and allocates at 4736: a smaller budget makes the tiling
search size the operation differently, so a mapping that fits sits below one that
does not. A bisection over that predicate returns a boundary rather than a floor,
and the tests below are what say the script knows the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import tight_budget_floor as floor  # noqa: E402


def monotone(threshold: int) -> floor.Allocates:
    """Allocates at or above `threshold` and nowhere below it."""

    def predicate(budget: int) -> bool:
        return budget >= threshold

    return predicate


def allocates_only_at(*budgets: int) -> floor.Allocates:
    """Allocates at exactly these budgets, which is as non monotone as it gets."""

    def predicate(budget: int) -> bool:
        return budget in budgets

    return predicate


def counting(inner: floor.Allocates, calls: list[int]) -> floor.Allocates:
    """The same predicate, recording every budget it was asked about."""

    def wrapped(budget: int) -> bool:
        calls.append(budget)
        return inner(budget)

    return wrapped


# ---------------------------------------------------------------------------
# The bisection.
# ---------------------------------------------------------------------------


def test_the_bisection_finds_the_floor_of_a_monotone_predicate() -> None:
    """The easy case, and the one ADR 0008's downward sweep also answers."""
    assert floor.bisect_floor(monotone(6464), 8192) == 6464
    assert floor.bisect_floor(monotone(64), 8192) == 64


def test_the_bisection_is_cheaper_than_the_sweep_it_replaces() -> None:
    """The reason for bisecting at all, stated as a number rather than a claim.

    ADR 0008 swept downward in 64 byte steps. From 194624 to a floor of 194240
    that is six compiles and from 194624 to 64 it is three thousand. The
    bisection is logarithmic in the range, and this pins that it stays so.
    """
    calls: list[int] = []
    assert floor.bisect_floor(counting(monotone(1024), calls), 194624) == 1024
    assert len(calls) < 40, calls


def test_a_ceiling_below_the_floor_is_refused_rather_than_answered() -> None:
    """A search for a floor under a ceiling that is itself below it has no answer.

    Returning the ceiling would be this script inventing a floor, which is the
    failure Section 16.1 forbids for a result field, arriving in a search.
    """
    assert floor.bisect_floor(monotone(8192), 6464) is None


# ---------------------------------------------------------------------------
# The window, and the descent it triggers.
# ---------------------------------------------------------------------------


def test_the_window_is_silent_on_a_monotone_predicate() -> None:
    assert floor.verify_window(monotone(6464), 6464) == []


def test_the_window_finds_the_budget_below_the_boundary_that_allocates() -> None:
    """`conv_bn_relu_stack`'s shape, written down as a predicate.

    Allocates above 4928 and also at 4736 and 4672, and nowhere between. A
    bisection stops at 4928 and calls it the floor; the window is what says it is
    only a boundary.
    """

    def predicate(budget: int) -> bool:
        return budget >= 4992 or budget in (4736, 4672)

    assert floor.bisect_floor(predicate, 6464) == 4992
    assert floor.verify_window(predicate, 4992) == [4736, 4672]


def test_the_descent_reports_the_smallest_budget_it_saw_allocate() -> None:
    """Exhaustive rather than bisected, because the predicate is not monotone."""
    assert floor.descend(allocates_only_at(4736, 4672, 4544), 4736) == 4544


def test_the_descent_stops_after_its_patience_and_does_not_run_to_zero() -> None:
    """A stopping rule rather than a proof, and it is bounded on purpose.

    The alternative is a search that walks to 64 bytes on every model that has a
    single non monotone point, which turns a check into an afternoon.
    """
    calls: list[int] = []
    predicate = counting(allocates_only_at(4736), calls)
    assert floor.descend(predicate, 4736) == 4736
    assert len(calls) == floor.PATIENCE_STEPS


def test_the_descent_never_returns_a_budget_it_did_not_see_allocate() -> None:
    """The property that makes the reported floor a measurement.

    Whatever the predicate does, the answer is a budget the predicate said yes
    to. A search that could return an untested value would be reporting a floor
    nobody stood on.
    """
    seen: list[int] = []

    def every_other_step(budget: int) -> bool:
        return budget % 128 == 0

    predicate = counting(every_other_step, seen)
    answer = floor.descend(predicate, 4736)
    assert predicate(answer)


# ---------------------------------------------------------------------------
# The constants.
# ---------------------------------------------------------------------------


def test_the_frozen_budgets_here_are_the_registry_s() -> None:
    """Two copies of ADR 0008's seven constants, held in step by a test.

    The script quotes them so its report can put the frozen and the measured
    floor side by side, and a quoted constant that drifts from the one the
    compiler uses would make every row of that report a comparison of two
    different things.
    """
    from npu_frontend import model_generator

    registry = {
        name: spec.tight_budget for name, spec in model_generator.MODELS.items()
    }
    assert registry == floor.FROZEN


def test_the_quantum_is_the_allocators_own_alignment() -> None:
    """64 bytes, and ADR 0008's reason for it rather than a page size.

    A row of the 16 by 16 array at f32 is 64 bytes, so it is the smallest step at
    which a budget change can move a placement at all. Stepping in anything
    coarser would quantise away the only two spilling cells the suite has, which
    is the deviation from Section 15 that ADR 0008 records.
    """
    assert floor.QUANTUM == 64
