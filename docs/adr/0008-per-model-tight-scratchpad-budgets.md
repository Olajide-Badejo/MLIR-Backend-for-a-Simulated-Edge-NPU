<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 8. The per model tight scratchpad budgets, measured at P8

- **Status:** Accepted
- **Date:** 2026-08-31, re-measured unchanged 2026-09-01
- **Diataxis type:** explanation

## Context

Section 15 asks for a tight scratchpad budget per model and gives the reason: a
single constant across structurally different models either fails to spill on
some, measuring nothing, or fails to allocate on others. It also gives the rule:
measure each model's peak from the allocated `npuisa.scratchpad_bytes` at the
default budget, then set the tight budget as a fixed fraction of that peak,
rounded to a multiple of 4096.

Section 15 also says when: P8, because P8 is the first phase at which
`npuisa.scratchpad_bytes` exists for every model at the default budget. Every
later phase's tight budget cells depend on these numbers, and a moved tight
budget moves every tight budget cell in the project's history at once, which is
a prime directive breach dressed up as a re-measurement.

**The measurement did not fit the rule, and the rule is the part that moves.**
Measured on 2026-08-31, the allocated peak at the default budget of 1048576
bytes:

| Model | Allocated peak | Sweep line peak |
|---|---|---|
| `lenet` | 194592 | 194560 |
| `depthwise_separable` | 8192 | 8192 |
| `resnet_block` | 8480 | 8480 |
| `inception_block` | 6848 | 6728 |
| `conv_bn_relu_stack` | 6432 | 6432 |
| `dilated_stack` | 8036 | 8008 |
| `lenet_batched` | 200800 | 200800 |

A fixed fraction of any of those, rounded to a multiple of 4096, does not
allocate. At 0.75 and 4096 byte rounding, every one of the seven is refused, and
the refusals are not fragmentation:

```
lenet, budget 143360: the scratchpad budget of 143360 bytes is too small: this
buffer of 192000 bytes could not be placed below offset 143360 in @main, and no
buffer live across the pressure peak can be spilled
```

**The mechanism is that the peak of these models is set by one instruction's own
operand set.** LeNet's largest fully connected layer holds a 400 by 120 weight
matrix, which is 192000 bytes, and it must be resident while the `MATMUL` that
reads it runs. Spilling it does not help: a spilled buffer is reloaded before
each use, so the reload is resident at the same moment the original would have
been. Any budget below that one buffer's footprint is unallocatable no matter
what the allocator spills, and Section 13 already names the remedy, which is
**tiling rather than spilling**. Tiling lands at P13.

Sweeping the budget in 64 byte steps and taking the smallest that allocates:

| Model | Smallest allocatable | As a fraction of the peak | Spills | Spill DMA |
|---|---|---|---|---|
| `lenet` | 194624 | 1.0002 | 0 | 0 |
| `depthwise_separable` | 8192 | 1.0000 | 0 | 0 |
| `resnet_block` | 6464 | 0.7623 | 1 | 3 |
| `inception_block` | 6144 | 0.8972 | 3 | 8 |
| `conv_bn_relu_stack` | 6464 | 1.0050 | 0 | 0 |
| `dilated_stack` | 8064 | 1.0035 | 0 | 0 |
| `lenet_batched` | 200832 | 1.0002 | 0 | 0 |

Five of the seven cannot go below their peak at all. Two can, because their
pressure comes from several concurrently live buffers rather than from one
instruction's operands, and those two spill.

A fraction above one is not a fraction. Rounding to 4096 makes it worse rather
than better: `inception_block` spills three buffers at 6144 and spills nothing at
8192, so the specification's own rounding quantum would take the one model with
the most interesting tight budget cell and turn it into a second copy of its
default budget cell.

## Decision

**The tight budget of each model is the smallest budget at which it allocates,
measured at 64 byte granularity on 2026-08-31 and frozen as a constant.**

| Model | Tight budget, bytes |
|---|---|
| `lenet` | 194624 |
| `depthwise_separable` | 8192 |
| `resnet_block` | 6464 |
| `inception_block` | 6144 |
| `conv_bn_relu_stack` | 6464 |
| `dilated_stack` | 8064 |
| `lenet_batched` | 200832 |

They live in `MODELS` in `python/npu_frontend/model_generator.py`, as a required
field of `ModelSpec` with no default, so a model added without one is an error at
import rather than a model quietly given somebody else's number.

**Two deliberate deviations from Section 15, and this is where they are
recorded.**

**The rounding quantum is 64 bytes and not 4096.** Sixty four is the allocator's
own alignment, which is a row of the 16 by 16 array at f32, so it is a number
this machine already has rather than one imported from a page size. On models
whose whole working set is six kilobytes, a 4096 byte quantum is a 64 percent
step and it quantises away the only two spilling cells the suite has.

**The fraction is recorded and is not operative.** It is 1.0 in effect on all
seven models, because the floor binds everywhere. Recording a fraction of 0.75
and then applying a floor that overrides it on all seven would be recording a
number that does nothing. The fraction becomes a live knob at **P13**, when
tiling gives the compiler a way to fit an instruction whose operands exceed the
budget, and the phase that lands tiling re-measures these constants once, in its
own commit, with `docs/BREAKING_CHANGES.md` written first per ground rule 7.

**Until then the constants above are frozen.** They are not recomputed on a run,
not derived from a sweep at test time, and not adjusted when the allocator
improves. A test asserts each one still allocates, which is the property the
project depends on; it does not assert each one is still the floor, because the
floor moving is exactly what an allocator improvement looks like and it must not
silently move the numbers.

## Consequences

**Two of the seven tight budget cells exercise spilling and five do not.**
`resnet_block` spills one buffer with three DMA operations, and its instruction
count goes from 14 to 17. `inception_block` spills three with eight, and goes
from 14 to 22. The other five compile to the same instruction stream at both
budgets, so their tight budget cells measure that the allocator is stable under
a budget change and nothing more. That is a real if modest thing to measure, and
saying so is better than implying seven spilling cells.

**Numerics do not move at the tight budget, and that is asserted rather than
assumed.** A spill is a DMA round trip, which copies bytes; the arithmetic is
untouched. Every model's outputs at its tight budget are bit identical to its
outputs at the default budget, checked over the input classes.

**The suite is small, and this record is what says so out loud.** A tight budget
that cannot be made tight is a statement about the models rather than about the
allocator. If a later phase wants tight budget cells that stress spilling on more
than two models, the answer is a model with a larger working set, not a smaller
constant, and the answer after P13 is tiling.

**The anchor model's tight budget is fixed.** Section 15 requires it, so that
LeNet's numbers stay comparable across the project's history. 194624 is that
number and it does not move without a `BREAKING_CHANGES.md` entry first.

## Re-measured on 2026-09-01, and nothing moved

*Recorded here rather than amended, because the decision did not change. This is
the second measurement of a frozen constant and the constant came out the same,
which is a fact worth having written down: the alternative is a later reader
finding a suite change in the history and having to wonder.*

Interphase P9b gave `dilated_stack`'s `conv1` a separate channel shaped bias
`Add`, so that `-npu-fuse-bias` has a target in Section 15's suite. Adding a node
to a model adds buffers, and the P9 handoff named that model's tight budget as
the thing the change might move. It was measured by this record's own sweep, at
64 byte granularity, at every level the compiler now builds rather than at `-O0`
alone:

| Model | Level | Allocated peak | Smallest allocatable | Frozen constant |
|---|---|---|---|---|
| `dilated_stack` | `-O0` | 8036 | 8064 | 8064 |
| `dilated_stack` | `-O1` | 8036 | 8064 | 8064 |
| `dilated_stack` | `-O2` | 8036 | 8064 | 8064 |

**Both numbers are the P8 ones**, so the constant stands and no
`BREAKING_CHANGES.md` entry moves it.

**The mechanism, because a measurement that came out unchanged still had a
reason.** The two buffers the new node adds are a twenty byte bias constant and a
360 byte destination, and both are live near the end of the program, where the
working set is a little over five kilobytes. This model's peak is set by `conv0`,
whose input, filter and result are resident together, and that instruction is
untouched. A sweep line peak is the maximum over time and not a sum over the
program, so a model can gain buffers and keep its peak, and this one did.

The six other constants were not re-measured and did not need to be: no other
model's graph changed, and the generator draws the new initializer after
`conv1.weight`, so every other tensor in the suite is bit identical.
