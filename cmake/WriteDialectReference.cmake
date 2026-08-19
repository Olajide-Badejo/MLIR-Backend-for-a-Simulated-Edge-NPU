# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The copy step of the npu-dialect-doc target. Run in script mode with
# -DGENERATED= and -DOUTPUT=.
#
# It exists as its own script rather than as a chain of COMMANDs because the
# committed file needs three things the raw mlir-tblgen output does not have: an
# SPDX header, so `reuse lint` stays clean; a line saying the file is generated
# and naming the command that regenerates it, so a reader who finds a mistake
# knows where to fix it; and a stable line ending policy, so a regeneration on a
# different machine is not a diff.
#
# Determinism matters here more than it looks. This file's whole purpose is to
# be diffed by CI, so anything in it that varies between two runs on the same
# input turns the staleness gate into a coin toss. Nothing below depends on the
# clock, the path, or the filesystem's iteration order.

if(NOT EXISTS "${GENERATED}")
  message(FATAL_ERROR
    "The generated dialect reference was not found at ${GENERATED}. "
    "The npu-dialect-doc target depends on NPUDialectReferenceDocGen, so this "
    "means the documentation generation itself did not run.")
endif()

file(READ "${GENERATED}" body)

set(header "<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

<!--
THIS FILE IS GENERATED. Do not edit it.

It is produced from the ODS source in include/NPU/Dialect/NPU/IR by

    ninja -C build npu-dialect-doc

and CI regenerates it and diffs, so an edit here is reverted by the next build
and reported as staleness by the next run. The place to change any of the prose
below is the operation's description in NPUOps.td, NPUAttrs.td or NPUTypes.td.
-->

")

file(WRITE "${OUTPUT}" "${header}${body}")
message(STATUS "Wrote ${OUTPUT}")
