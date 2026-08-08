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

None yet. The baseline was recorded at the start of the upgrade work and has not
been deliberately broken since.
