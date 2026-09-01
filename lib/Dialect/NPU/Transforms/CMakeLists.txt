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
        ConstantFold.cpp
        FoldBatchNorm.cpp
        FuseBias.cpp
        FuseOps.cpp

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
        )
