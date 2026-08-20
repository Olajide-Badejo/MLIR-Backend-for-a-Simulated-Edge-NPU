# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The transformation library is separate from the dialect library on purpose.
# MLIRNPUISA is linked into everything that parses this IR, including the future
# encoder and simulator front doors, and it must not drag the dialect conversion
# infrastructure in with it. A pass library that depends on the conversion stack
# and a dialect library that does not is the split MLIR itself uses, and it is
# what keeps the dependency on MLIRTransforms and MLIRTransformUtils on the side
# of the build that actually rewrites IR.

add_mlir_dialect_library(MLIRNPUISATransforms
        AllocateScratchpad.cpp
        LowerNPUToNPUISA.cpp
        ScratchpadAllocation.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPUISA/Transforms

        DEPENDS
        MLIRNPUISATransformsIncGen

        LINK_LIBS PUBLIC
        MLIRNPU
        MLIRNPUISA
        MLIRArithDialect
        MLIRFuncDialect
        MLIRIR
        MLIRMemRefDialect
        MLIRPass
        MLIRSupport
        MLIRTensorDialect
        MLIRTransforms
        MLIRTransformUtils
        )
