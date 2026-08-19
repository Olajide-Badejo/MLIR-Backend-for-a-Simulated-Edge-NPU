# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# A separate library from MLIRNPU on purpose. This one links the tiling stack
# and the tensor dialect; MLIRNPU does not, and every tool that only needs to
# parse and verify npu IR links MLIRNPU alone.

add_mlir_dialect_library(MLIRNPUTilingInterfaceImpl
        NPUTilingInterfaceImpl.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPU/Interfaces

        DEPENDS
        MLIRNPUOpsIncGen

        LINK_LIBS PUBLIC
        MLIRNPU
        MLIRIR
        MLIRSupport
        MLIRTensorDialect
        MLIRTilingInterface
        )
