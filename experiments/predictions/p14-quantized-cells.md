<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: the 63 quantized cells, and the suite at 280

- **id:** p14-quantized-cells
- **written:** 2026-10-06
- **result field:** every field of the 63 quantized cells, every field of the
  217 fp32 cells against `d2caa05`, the suite's count and its measured cost
- **direction:** the quantized cells reproduce what item 4 and the end to end
  test already measured; nothing an fp32 cell measured moves
- **magnitude bracket:** exact where the arithmetic is the same arithmetic; the
  suite at 280 cells in 5.5 to 8 minutes
- **answered at:** P14, the record that follows the commits the 2026-10-06
  declaration in `docs/BREAKING_CHANGES.md` names

*Written and committed after the declaration and before any of its code exists.
The inputs are committed or already measured: item 4's measurement of the same
63 configurations at `387a641`, recorded in `docs/ENGINEERING_LOG.md` under
2026-09-30, the quantized end to end test's figures, and the 217 committed fp32
cells.*

## Hypothesis

### 1. The count

The driver plans 280 cells: 63 fp32 benchmark cells, 63 quantized ones and 154
fp32 ablation cells. Nothing in the count is written by hand; the quantized half
is the fp32 benchmark grid with `quantized` set.

### 2. The fp32 cells do not move

All 217, re-recorded in the same run, carry the same value in every field they
carried at `d2caa05`, apart from the timing objects, the timestamps, the git
sha, the content hash and `manifest.run_order_position`. **The last moves on
nearly all of them and on nothing else**: the seeded shuffle over 280 cells is
a different permutation from the one over 217. The fp32 cells' prediction ids
do not move.

### 3. The quantized cells reproduce what was already measured

- **Cycles, cycles without the packing, and energy**: equal, on every one of
  the 63, to item 4's measurement of the same configuration, to the last bit.
  The program is the same program and the charge and the coefficients are the
  same ones.
- **`int8_macs` equals `macs`** on every quantized cell, and
  **`quant_boundary_crossings`** is the end to end test's count at each level:
  32 over the seven at `-O0` and `-O1` and 28 at `-O2`, per model 6, 2, 4, 6,
  6, 6, 2 and 4, 2, 4, 4, 6, 6, 2, at both batches and both budgets, because a
  crossing is an instruction and neither the batch nor the budget adds one.
- **`max_abs_error_vs_onnxruntime`** at the default budget and the declared
  batch equals the end to end test's largest error, because the inputs and the
  program are the same: 0.0061434, 0.00058782, 0.046067, 0.023259, 0.0011986,
  0.0012821 and 0.012631 at `-O0` and `-O1`, and 0.0043167 for
  `conv_bn_relu_stack` and 0.032906 for `dilated_stack` at `-O2`.
- **`sqnr_db_vs_fp32_simulated`**, against the fp32 twin at the same model,
  level, budget and batch, is within 0.01 dB of the end to end test's SQNR
  against onnxruntime at the declared batch, because the fp32 twin sits within
  parts in ten million of onnxruntime: 33.80, 52.68, 34.40, 36.29, 45.24, 46.76
  and 47.83 dB at `-O0`, and 34.75 and 36.52 dB for the two that move at
  `-O2`. **`max_abs_error_vs_fp32_simulated`** is within 1e-6 of
  `max_abs_error_vs_onnxruntime` on every quantized cell, for the same reason.
- **The budgets agree**: a quantized cell at the tight budget computes the same
  answer, bit for bit, as at the default budget at the same batch and level.

### 4. The roofline and SCALE-Sim on an int8 program

- **No quantized cell is below its roofline**, with each contraction bounded at
  the peak its arithmetic runs at.
- **SCALE-Sim's cycles are the fp32 twin's**, exactly, on every quantized cell
  whose fp32 twin's contractions are not tiled, because the topology is the same
  shapes and SCALE-Sim models no data width. Where the fp32 twin tiles, at the
  tight budget at `-O2`, they differ, because the int8 program declines the
  tiling.
- **The `int8_packing` term is exactly minus three times the int8 contractions'
  kernel charge** on every quantized cell and zero on every fp32 one, and the
  decomposition still sums with a zero residual.

### 5. The cost

A quantized cell compiles through one more pass and runs its program twice, so
it costs more than an fp32 cell: **280 cells in 5.5 to 8 minutes, 1.2 to 1.7
seconds a cell**, inside Section 2's 90 minute budget.

## What would falsify it

- A count other than 63, 63 and 154, or one written by hand.
- Any fp32 field outside the five named moving, or a prediction id moving.
- Any quantized cell's cycles, cycles without packing or energy differing from
  item 4's by any bit, or `int8_macs` differing from `macs`, or a crossing count
  other than the table's.
- A largest error at the declared batch differing from the end to end test's;
  an SQNR more than 0.01 dB from it; the two error figures more than 1e-6 apart.
- A quantized answer that differs between the budgets.
- A quantized cell below its roofline; SCALE-Sim cycles differing from an
  untiled fp32 twin's; an `int8_packing` term other than minus three times the
  kernel charge, or a nonzero residual.
- The suite outside 5.5 to 8 minutes.
