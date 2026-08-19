# Breaking changes

Every entry here is a deliberate regression against the recorded baseline in
`test/baseline/baseline.json`.

The upgrade work runs under a prime directive: do not break working behaviour,
and if you must, say so explicitly, in writing, before you do it. This file is
where that gets said. `scripts/regression-baseline.sh --check` runs at every
phase gate and fails on any drift in test outcomes, instruction counts,
simulated cycles, DRAM traffic, end to end error, or the golden output tensors.
When a change moves one of those on purpose, the fix is not to re-record the
baseline quietly. It is to write the change down here first, then re-record.

An entry must say what moved, by how much, why the new behaviour is the correct
one, and which commit did it. Numbers, not adjectives. If a golden tensor moved,
give the max absolute delta. If a cycle count moved, give both figures.

Entries are newest first. `CHANGELOG.md` carries the user facing summary; this
file carries the baseline accounting behind it.

## Format

```
## <date>: <one line summary>

**Moved:** which baseline fields changed, with before and after numbers.

**Why this is correct:** the reasoning. A regression that is genuinely a fix
should say what was wrong with the old number.

**Commit:** <sha>
```

## Entries

## 2026-08-09: two end to end test names removed, superseded by the matrix

**Moved:** the `pytest` suite's recorded test name list loses two entries:

- `test.Python.test_end_to_end::test_lenet_matches_onnxruntime`
- `test.Python.test_end_to_end::test_lenet_with_spilling_matches_onnxruntime`

No instruction count, cycle count, DRAM figure, end to end error, or golden
tensor moved. The pytest suite grows from 41 to 72 collected tests, so this is a
rename and expansion rather than a reduction in coverage, but the baseline
records test names and two of the recorded names no longer exist, which it
correctly reports as drift.

**Why this is correct:** both tests are replaced by cells of the new
parametrized matrix in `test/Python/test_end_to_end.py`, and the replacement is
strictly larger. `test_lenet_matches_onnxruntime` compiled with a hardcoded pass
list and checked one standard normal draw at the 1 MB budget; that cell is now
`test_matrix_cell[lenet-O2-1048576-normal]`, and the matrix additionally covers
`-O0` and `-O1`, the 140 KB budget, and four further input classes.
`test_lenet_with_spilling_matches_onnxruntime` is now
`test_matrix_cell[lenet-O2-143360-normal]` plus the same expansion.

The old tests were also validating a pipeline nobody can request: their pass
list, `-canonicalize -npu-fuse-ops -npu-lower-to-npuisa
-npu-allocate-scratchpad`, is `-O2` without the second `-canonicalize` and
without `-symbol-dce`, so it matches no `-O` level. Keeping the names alive
would have meant keeping that fiction alive alongside the matrix.

Coverage of the two original cells was confirmed before the deletion rather than
asserted after it: `test_matrix_covers_the_full_cross_product` fails if any cell
id is missing from the collected set.

**Commit:** `4fda494`
