<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: what the cost model's INT8 terms move

- **id:** p14-int8-cost-terms
- **written:** 2026-09-30
- **result field:** the int8 MAC coefficient and its ratio to the published
  figure; `simulated_cycles` and `simulated_cycles_without_int8_packing` of the
  quantized programs; their energy per inference and its split by component;
  every field of the 217 fp32 cells and of the baseline
- **direction:** nothing fp32 moves; the quantized cycles do not move; the
  quantized energy falls to well under the fp32 energy on every model
- **magnitude bracket:** an int8 MAC at 1.0025 pJ, 4.36 times the published
  8 bit multiply plus add; the throughput assumption's share of the cycle win
  per model in the table below; the quantized energy between 0.30 and 0.65 of
  the fp32 energy at the default budget and between 0.05 and 0.30 at the tight
  one
- **answered at:** P14, the record that follows the commits the 2026-09-30
  declaration in `docs/BREAKING_CHANGES.md` names

*Written and committed after the declaration and before any of the code it
declares exists. The inputs are committed or already measured: the 217 fp32
cells, the baseline, the plug in's own tables at its pinned sha, and the
quantized cycles measured at `a6d80b8` at the calibration's new position, which
nothing in this item changes.*

## Hypothesis

### 1. The int8 MAC coefficient

The `Aladdin_table` plug in at `5e2e126` answers an integer multiplier from its
32 bit, 1 ns row, 12.68 pJ, scaled by the square of the width, and an integer
adder from its 32 bit, 1 ns row, 0.21 pJ, scaled linearly. An 8 bit multiplier
is therefore 12.68 / 16 = 0.7925 pJ and a 32 bit adder 0.21 pJ, so **an int8 MAC
is 1.0025 pJ**, exactly, at Accelergy's six significant figures. That is
**1/49.2 of the fp32 MAC's 49.286 pJ**.

Section 16.4's published figures are an 8 bit integer multiply at about 0.2 pJ
and an 8 bit integer add at about 0.03 pJ, 0.23 pJ together. **The ratio is
4.36, inside an order of magnitude, so the check passes**, where the fp32 MAC's
10.71 did not. It is high for a reason that can be named in advance: the
machine accumulates in 32 bits and the published add is an 8 bit one, and the
table is a synthesised unit at a 1 ns clock rather than a combinational
datapath, which is the same cause the fp32 figure has.

The two int8 subcomponents carry an area scale of zero. **The area a machine
with four separate int8 MAC units per lane would add is between 0.3 and 1.5
mm2** over the 256 lanes, against the fp32 array's 2.129 mm2.

### 2. Nothing fp32 moves

The int8 MAC action is counted by `int8_macs`, which is zero on every fp32
cell, and the fp32 MAC's action and coefficient are untouched; the new
subcomponents add no area; the cycle option is off unless asked for. So **all
217 fp32 cells re-recorded carry the same value in every field they carried
before**, apart from the timing objects, the timestamps, the git sha and the
content hash, and gain only `simulated_cycles_without_int8_packing` as a null
with its reason and `I8_MACS_PER_LANE` in the manifest's constants. The
baseline's cells, suites apart, do not move; its `energy_per_action_pj` gains
`mac_array.int8_mac` at 1.0025 pJ at each of its seven budgets, and nothing in
it moves.

### 3. The quantized cycles do not move

The integer charge is not touched, so every quantized program takes exactly the
cycles it took at `a6d80b8`: 1156, 866.5, 950.9375, 1511, 5407.5, 8734.375 and
1132 at `-O0` and the default budget at each model's declared batch, and the
same at `-O1`; at `-O2`, 876 for `conv_bn_relu_stack` and 931.6875 for
`dilated_stack`, and the `-O0` figure for the other five.

### 4. The throughput assumption's share of the cycle win

The quantized program's cycle win over its fp32 twin splits into two parts: the
packing's, `simulated_cycles_without_int8_packing - simulated_cycles`, and the
rest, which is the DMA traffic reduction net of what the crossings cost. Call
the packing's share of the whole win `phi`.

**The programs at batch 1 are small, and their fp32 twins are closer to compute
bound than to DMA bound**, apart from LeNet, whose fully connected weights make
it DMA bound at 16441 DMA cycles against 7956 compute. Everywhere else the
int8 weights save a few hundred bytes behind a 64 cycle descriptor, which buys
little, while the packing divides the contractions' compute by four. And the
quantized program carries `QUANT` and `DEQUANT` instructions the fp32 one does
not, each with its issue overhead, so **without the packing the quantized
program can be slower than its fp32 twin, which is `phi` above 1**: the packing
then pays for the whole win and for the crossings as well.

| Model, `-O0`, default budget, declared batch | fp32 cycles | quantized cycles | predicted `phi` |
|---|---|---|---|
| `conv_bn_relu_stack` | 1372.5 | 1156 | 0.5 to 2.0 |
| `depthwise_separable` | 1324 | 866.5 | 0.6 to 1.3 |
| `dilated_stack` | 1243.6875 | 950.9375 | 0.5 to 1.5 |
| `inception_block` | 2398.5 | 1511 | 0.6 to 1.3 |
| `lenet` | 17766.25 | 5407.5 | **0.1 to 0.45** |
| `lenet_batched`, batch 4 | 20000 | 8734.375 | 0.6 to 1.1 |
| `resnet_block` | 1626 | 1132 | 0.6 to 1.3 |

`-O1` is `-O0`'s program on every model, so the same. At `-O2` the five whose
program does not change keep `-O0`'s `phi` wherever their fp32 twin's cycles
are `-O0`'s too, and `conv_bn_relu_stack` and `dilated_stack` fall inside the
same brackets. **At batch 4 a model's contractions grow four times while its
weights are fetched once**, and LeNet is where that shows: its `phi` at batch 4
is in `lenet_batched`'s bracket, 0.6 to 1.1, because at batch 4 the two are the
same program, so the DMA reduction is most of LeNet's win at batch 1 and a
minority of it at batch 4. The other five are in 0.5 to 1.5 at batch 4. The
simulated cycles without packing are never below the cycles with it, on any
configuration.

### 5. The quantized energy

The fp32 energy of these programs is 37 to 79 percent the array's at the
default budget, and the rest is the 1 MB scratchpad, at 127.7 pJ a read, and
DRAM. Quantized, the array's share becomes almost nothing, the DRAM share falls
with the int8 weights, and the scratchpad's accesses are about the same or
more, because every crossing is a `QUANT` or `DEQUANT` that reads and writes
its tensor once more at the scratchpad port while each fused relu reads and
writes it once less.

- **The quantized energy per inference is between 0.30 and 0.65 of its fp32
  twin's at the default budget** on every model and level, and LeNet, with the
  largest DRAM share, is at the low end of that range.
- **At a tight budget between 0.05 and 0.30**, because a small scratchpad costs
  6.4 pJ a read rather than 127.7, which leaves the fp32 energy array
  dominated and the quantized energy with little in it.
- **The array is below 5 percent of every quantized program's energy, and the
  scratchpad is its largest consumer at the default budget on every model.**

## What would falsify it

- Any fp32 field of the 217 cells, or any baseline field other than the added
  coefficient, moving; or any golden moving.
- An int8 MAC other than 1.0025 pJ, or a ratio outside an order of magnitude.
- Any quantized program's cycles other than the figures in section 3.
- A `phi` outside its bracket, at either batch, or a configuration whose cycles
  without packing are below its cycles with it.
- A quantized energy outside its bracket, an array share at or above 5 percent,
  or a model whose largest quantized consumer at the default budget is not the
  scratchpad.
