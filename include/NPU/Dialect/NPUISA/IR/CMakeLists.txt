# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The TableGen wiring for the npuisa dialect, the same shape as the npu one next
# door.

# add_mlir_dialect generates the op, dialect and type declarations and creates
# the MLIRNPUISAOpsIncGen target the library DEPENDS on. Unlike the npu dialect
# this one does define a type, so the type half of what this command generates
# is used rather than inert.
add_mlir_dialect(NPUISAOps npuisa)

# The dialect reference, regenerated as a build target so it cannot drift from
# the source. The npuisa reference is written into the same committed file as
# the npu one by the npu-dialect-doc target at the top level, which is what
# keeps one document describing both dialects rather than two documents nobody
# reads twice.
add_mlir_doc(NPUISAOps NPUISADialectReference NPUISA/ -gen-dialect-doc
             -dialect=npuisa)
