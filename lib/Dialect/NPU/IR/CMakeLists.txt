# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The shape here follows mlir/examples/standalone/lib/Standalone/CMakeLists.txt.
#
# Note what is NOT linked: nothing from the SCF or tensor tiling stack. The
# TilingInterface implementation lives in its own library next door precisely so
# that this one, which every tool in the project links, does not carry that
# dependency.

add_mlir_dialect_library(MLIRNPU
        NPUAttrs.cpp
        NPUDialect.cpp
        NPUOps.cpp
        NPUShapeUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPU

        DEPENDS
        MLIRNPUOpsIncGen
        MLIRNPUOpsAttrsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRBytecodeOpInterface
        MLIRControlFlowInterfaces
        MLIRDestinationStyleOpInterface
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
        MLIRTilingInterface
        )
