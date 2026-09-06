# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The npu dialect's own transformations, separate from MLIRNPU for the same
# reason MLIRNPUISATransforms is separate from MLIRNPUISA: the dialect library
# is linked into everything that parses this IR, including the encoder and the
# simulator front doors, and it must not drag the rewrite infrastructure in with
# it.

add_mlir_dialect_library(MLIRNPUTransforms
        AssignLayout.cpp
        ConstantFold.cpp
        FoldBatchNorm.cpp
        FuseBias.cpp
        FuseOps.cpp
        TileToScratchpad.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPU/Transforms

        DEPENDS
        MLIRNPUTransformsIncGen

        LINK_LIBS PUBLIC
        MLIRNPU
        MLIRFuncDialect
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRTensorDialect
        MLIRTransforms
        MLIRTransformUtils
        # -npu-tile-to-scratchpad consumes TilingInterface through the upstream
        # scf tiling driver and unrolls the loops it generates, and it scores
        # candidates with the simulator's own cost model rather than a second
        # copy of it. Both dependencies are the point rather than an accident:
        # Section 5.5 puts the constants in one home and says the tiling pass is
        # what scores against it. -npu-assign-layout reads the same home, for
        # the same reason: its whole decision is one term out of that model.
        MLIRSCFDialect
        MLIRSCFTransforms
        MLIRSCFUtils
        MLIRTilingInterface
        NPUCostModel
        )
