<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 10. The tight budget is a property of a program, so it does not cross the batch axis

- **Status:** Accepted
- **Date:** 2026-09-01
- **Diataxis type:** explanation

## Context

Section 17.4's matrix sweeps the scratchpad budget and the batch size as two
independent axes, which reads as two budgets times two batches. P10 is the first
phase to build a cell for every point of that product, and the product does not
exist.

ADR 0008 defines the tight budget by measurement, not by formula: it is the
smallest budget in a 64 byte sweep at which the model still allocates, taken from
the allocated `npuisa.scratchpad_bytes` at the default budget. That definition is
about **a program**. `TIGHT_BUDGETS` is keyed by model, and each entry was
measured at that model's own declared batch, which is one for six of the seven
and four for `lenet_batched`.

A model at batch 4 is a different program. Measured on 2026-09-01, at `-O0` and
the default budget of 1048576 bytes:

| Model | Declared batch | Peak at declared batch | Peak at batch 4 | Recorded tight budget | Allocates at batch 4 |
|---|---|---|---|---|---|
| `lenet` | 1 | 194592 | 200800 | 194624 | no |
| `depthwise_separable` | 1 | 8192 | 32768 | 8192 | no |
| `resnet_block` | 1 | 8480 | 26912 | 6464 | no |
| `inception_block` | 1 | 6848 | 24576 | 6144 | no |
| `conv_bn_relu_stack` | 1 | 6432 | 18720 | 6464 | no |
| `dilated_stack` | 1 | 8036 | 32080 | 8064 | no |
| `lenet_batched` | 4 | 200800 | 200800 | 200832 | yes |

**None of the six models declared at batch 1 allocates at batch 4 under its
recorded tight budget.** The refusal is the one ADR 0008 already diagnosed and
its wording is unchanged:

```
the scratchpad budget of 8064 bytes is too small: this buffer of 16016 bytes
could not be placed below offset 0 in @main, and no buffer live across the
pressure peak can be spilled. The sweep line peak is 32032 bytes, which is a
lower bound on any placement
```

That is not fragmentation and not an allocator limitation. It is a single buffer
larger than the entire budget, and spilling cannot help because a spilled buffer
is reloaded before each use and the reload is resident at the same moment the
original would have been. ADR 0008 names the remedy and the phase: tiling rather
than spilling, at P13.

`lenet` is worth reading separately, because it is the one case where the
intuition "the peak scales with the batch" is wrong. Its peak barely moves, from
194592 to 200800, because it is set by a 400 by 120 weight matrix that no batch
size touches. It still fails, by six kilobytes, and that is the clearest possible
demonstration that the tight budget is a measured property of one program rather
than a formula anybody can extrapolate.

**ADR 0008's own procedure was re-run before this record was written, and it
reproduces every one of its seven constants exactly.** A 64 byte bisection for
the smallest allocatable budget, per model and per batch, returns 194624, 8192,
6464, 6144, 6464, 8064 and 200832 at each model's declared batch, which is ADR
0008's table to the byte. That matters twice over: it says the rule in ADR 0008
is reproducible a phase later on a changed model suite, and it says the batch 4
failures above are the rule working rather than the rule breaking. The same
sweep gives the batch 4 floors, which are recorded here as measurements and
deliberately **not** promoted to constants: `lenet` 200832, `depthwise_separable`
32768, `resnet_block` 24576, `inception_block` 24576, `conv_bn_relu_stack` 18752,
`dilated_stack` 32128.

One of those numbers is a finding in its own right and belongs to P13 rather
than here: `inception_block` spills three buffers at batch 1 and **none** at
batch 4. Its batch 1 tight budget of 6144 is below its peak, which is the case
ADR 0008 called the interesting one; its batch 4 floor of 24576 is at the peak,
so the spilling disappears exactly where a naive reading would expect more of it.

## Decision

**A cell that names the tight budget runs at the model's declared batch. A cell
that names the default budget runs at both batches.**

For the benchmark suite of `experiments/run_benchmarks.py` that gives 7 models
times 3 levels times 3 budget and batch combinations, which is 63 benchmark
cells, plus the ablatable `-O2` set times 7 models times 2 budgets at the
declared batch. **This decision fixes the 63 and nothing else**: the ablation
half is however many ablatable passes the driver reports, which was 8 and 112
when this record was written and is 11 and 154 from P13, for 217 in total. The
number this ADR is about is the 63.

**No constant in ADR 0008 moves and none is added.** The alternative considered
and rejected was to extend the 64 byte sweep to batch 4 and record seven more
constants. That is a re-measurement of the tight budgets, `docs/PHASE_STATE.md`
hands re-measurement to P13, and P13 is also the phase that makes a budget below
the peak reachable at all. Doing it here would mean P10 recording constants that
P13 immediately invalidates, and every tight budget cell measured against them
would have to be thrown away.

The other alternative, scaling the batch 1 budget by the batch size, was rejected
outright: the `lenet` row above shows the scaling factor is not the batch, and a
constant derived from a relationship this project has just measured to be false
is exactly the unverifiable number ground rule 1 forbids.

## Consequences

- The benchmark half of the suite has 63 cells rather than the 84 a free cross
  product would give. The count is computed from this rule rather than written
  down, so it moves when the rule or the model suite does, and it did not move
  at P13 when the ablatable set went from 8 to 11 and the suite from 175 cells
  to 217.
- Every ablation row's `baseline_cell` is a cell the same run measured, because
  ablation cells and their baselines share the declared batch. `fill_deltas`
  refuses a row whose baseline is missing, so this is checked rather than assumed.
- `lenet_batched`'s tight cell at batch 4 is a real tight cell. Its counterpart
  at batch 1 would not have been: 200832 bytes is generous for a batch 1 program
  whose peak is 194592, so the cell would have been a second copy of the default
  budget cell wearing the tight label. Excluding it is not only about the six
  that fail.
- **P13 inherits the question, and it inherits it stated rather than open.** When
  tiling lands, a tight budget below the peak becomes reachable and the sweep can
  be run per program rather than per model. At that point this decision is
  superseded rather than worked around, and the tight budget cells of every
  earlier phase are re-recorded in their own commit under ground rule 7.
- Section 17.4's pytest matrix is unaffected. It does not sweep the budget axis,
  for the reason its own module docstring gives: a tight budget changes where the
  buffers go and not what the arithmetic is, and `test/Python/test_tight_budgets.py`
  asserts that rather than assuming it.
