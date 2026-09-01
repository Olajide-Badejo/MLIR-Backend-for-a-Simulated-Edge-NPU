# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The pass declarations and the registration hooks, generated from Passes.td.
# The -name argument prefixes the generated registration function, so it is the
# dialect's name: registerNPUPasses is the entry point npu-opt calls, beside
# registerNPUISAPasses.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NPU)
add_public_tablegen_target(MLIRNPUTransformsIncGen)

add_mlir_doc(Passes NPUPasses ./ -gen-pass-doc)
