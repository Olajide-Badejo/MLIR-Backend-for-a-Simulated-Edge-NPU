# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# MLIRNPU is a PUBLIC link dependency rather than a duplicated set of attribute
# definitions. The two memory space attributes this dialect's memrefs are built
# on, #npu.scratchpad and #npu.dram, are defined once by the npu dialect, and a
# second definition would give one machine's one scratchpad two spellings that
# would then have to be kept equal by hand.
#
# MLIRMemRefDialect is here for the overlap analysis, which walks memref.view
# and memref.subview back to their allocations. That is a dependency on the
# builtin memref dialect, not on the tiling stack, and it is unavoidable: an
# analysis of views has to know what a view is.

add_mlir_dialect_library(MLIRNPUISA
        NPUISADialect.cpp
        NPUISAMemoryOverlap.cpp
        NPUISAOps.cpp
        NPUISATypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPUISA

        DEPENDS
        MLIRNPUISAOpsIncGen

        LINK_LIBS PUBLIC
        MLIRNPU
        MLIRIR
        MLIRSupport
        MLIRBytecodeOpInterface
        MLIRDestinationStyleOpInterface
        MLIRMemRefDialect
        MLIRSideEffectInterfaces
        )
