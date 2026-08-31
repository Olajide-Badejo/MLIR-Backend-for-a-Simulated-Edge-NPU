<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Breaking changes

*Diataxis type: reference.*

This file records every **deliberate** regression of the recorded baseline, and
it records each one **before** the commit that causes it.

The prime directive of the build specification is that once a behaviour is in
the baseline it does not change silently, and that if it must change I say so in
writing first. This file is where that writing goes. From Phase P8 the
repository carries a baseline of test names and counts, instruction counts,
simulated cycles, DRAM bytes and golden output tensors, and every phase gate
re-runs it. A gate that finds the baseline moved fails, and the only thing that
turns that failure into a pass is an entry here that predicted the movement.

The ordering is the whole mechanism. An entry written after the number moved is
an explanation; an entry written before it moved is a decision. Git commit order
is what tells the two apart, so the entry lands in its own commit, strictly
before the commit that changes the behaviour.

Two things do not belong here. A number that moved by accident is a defect and
goes in `DEFECT_LOG.md` until it is understood. A user visible change that does
not regress the baseline is a changelog line and goes in `CHANGELOG.md`. Some
changes are both, and then they are written in both places rather than in
whichever one was closer to hand.

A baseline field that did not exist yet cannot have regressed. The baseline
grows across phases with a `schema_version` bump each time, and the arrival of a
new field is not a breaking change.

## Entry form

Each entry names the date, the phase, which baseline fields move and in which
direction, roughly how far, why the regression is worth taking, and the commit
that causes it once it exists.

## Entries

None yet. The baseline does not exist before Phase P8, so nothing can have
regressed against it.
A sentence with an em dash — in it.
