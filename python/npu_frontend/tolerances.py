# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The end to end tolerances of Section 17.4, in one place.

*Moved here at P9b from `test/Python/test_end_to_end.py`, where they were
declared and where they are still used.* The move is D-0039's second half: the
regression baseline needs the same band the matrix enforces, `scripts/` cannot
import a test module, and a second copy of a tolerance constant is exactly the
duplication `test/Python/test_tool_discovery.py` exists to hunt for. One rule,
one place, two importers.

**What these bound.** The distance between an answer this compiler produced and
an answer a reference produced for the same graph and the same input:
`onnxruntime` for the whole chain, `refexec` for everything below the importer.
They do **not** bound the distance between two runs of this compiler. That is a
different question with a different answer, `GOLDEN_TOLERANCE` in
`scripts/regression_baseline.py`, and it is **zero**. Keeping the two apart is
the whole of Section 17.6's two bands, and P9b is the phase that found out what
happens when one of them is applied to the other's quantity.

## The measurement the bounds were set from

Measured on 2026-08-31 over the whole matrix, on this toolchain:

    against onnxruntime   worst absolute 4.77e-06   worst relative 8.08e-07
    against refexec       worst absolute 4.77e-06   worst relative 4.17e-07

Both worst cases are the `dilated_stack` and `depthwise_separable` cells at the
two constant classes, whose answers have magnitudes in the tens and whose
convolutions accumulate a few hundred same signed terms. That is where a
summation order difference is largest, and a summation order difference is what
these numbers are: the reference sums whole tensor slices per kernel position,
the simulator walks one output element at a time, and neither is wrong.

## Why the bounds are ten and six times the observed maxima

Not two, and the reason is stated rather than left as taste: **this suite runs
on more than one host and `onnxruntime` chooses its own vectorisation per host.**
A bound two times the observed value on one machine is a bound that goes red on
another for a reason that is not a defect. Ten is far below the 1e-3 against
3e-8 that Section 17.4 names as the tolerance that cannot fail.

**That sentence was written at P8 and P9b proved it with numbers.** The
`regression-baseline --check` step's activation proof runs landed on GitHub
runner hardware different from the two before them, and eighteen cells reported
a different distance to `onnxruntime`, moving between 1e-8 and 1e-7 **in both
directions**, with no golden tensor and no cycle count moving at all. The
simulator was bit identical; the oracle was not. So the per host caveat above is
not a hedge, it is a measured property of the reference implementation, and any
comparison of a distance to `onnxruntime` has to carry a band wide enough to
hold it.

## Never loosened to make a cell pass

If a cell needs a wider bound, that is a finding: record the measured value and
say why the bound moved. Neither number has moved since it was set.
"""

from __future__ import annotations

from typing import Final

#: The absolute band on the distance between this compiler's answer and a
#: reference's, for the same graph and input.
ABSOLUTE_TOLERANCE: Final[float] = 5e-5

#: The relative band on the same distance. Checked separately from the absolute
#: one and never combined into `|a - b| <= atol + rtol * |b|`, which is a single
#: predicate in which a large `atol` hides a failed relative bound and a large
#: `rtol` hides a failed absolute one.
RELATIVE_TOLERANCE: Final[float] = 5e-6

# ---------------------------------------------------------------------------
# The roofline slack of Section 16.6, added at P11.
#
# A different quantity from the two above and kept apart from them for the
# reason this module was created: two bands with different meanings sharing one
# name is how one gets applied to the other's quantity. These bound a distance
# between two answers; the one below bounds nothing but floating point rounding.
# ---------------------------------------------------------------------------

#: How far below its bound a cycle count may sit and still be read as at the
#: bound rather than under it.
#:
#: **Derived from the representation, not chosen.** Both sides of the comparison
#: are IEEE 754 doubles accumulated over the tiles of one instruction. The
#: largest tile count in this suite is `lenet`'s 400 by 120 matmul, which folds
#: into 25 times 8 tiles, so a few hundred additions rather than a few thousand;
#: taking 10^4 as a generous ceiling and 2.22e-16 as the unit roundoff gives a
#: relative error bound near 2.2e-12. This is 1e-9, three orders above that
#: ceiling and six orders below the smallest divergence that would mean
#: anything: a pass that produced cycles genuinely below a physical bound would
#: be under it by percent, not by parts per billion.
#:
#: **It is a slack on rounding and never on a finding.** Section 16.6 fails the
#: run on a cell below its bound. If a cell ever needs this widened to pass,
#: that is the finding, and widening it would be deleting the finding.
ROOFLINE_RELATIVE_SLACK: Final[float] = 1e-9
