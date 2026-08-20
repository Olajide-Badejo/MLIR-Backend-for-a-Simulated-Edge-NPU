# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The pass declarations and the registration hooks, generated from Passes.td.
# The -name argument is what prefixes the generated registration functions, so
# it is the dialect's name rather than the project's: registerNPUISAPasses is
# the entry point npu-opt calls.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NPUISA)
add_public_tablegen_target(MLIRNPUISATransformsIncGen)

add_mlir_doc(Passes NPUISAPasses ./ -gen-pass-doc)
