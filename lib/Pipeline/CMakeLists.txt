# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# add_mlir_library rather than add_mlir_dialect_library: this is not a dialect
# and it must not land in the MLIR_DIALECT_LIBS global property, which
# npu-opt reads to build its link line. A pipeline library appearing there
# would be linked into every tool that registers all dialects, including tools
# that have no pipeline to run.

add_mlir_library(MLIRNPUPipeline
        Pipeline.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Pipeline

        LINK_LIBS PUBLIC
        MLIRNPUISATransforms
        MLIRFuncDialect
        MLIRIR
        MLIRPass
        MLIRSupport
        )
