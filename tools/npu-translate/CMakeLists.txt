# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# npu-translate links the bridge library and the dialects it parses. It does
# not link MLIROptLib or the whole of MLIR's pass registry, because it runs no
# passes: it takes IR that -npu-lower-to-npuisa and -npu-allocate-scratchpad
# have already produced and turns it into a file.

set(LIBS
        MLIRNPUEncoding
        NPUEncoding
        MLIRNPU
        MLIRNPUISA
        MLIRArithDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRParser
        MLIRIR
        MLIRSupport
        )

add_llvm_executable(npu-translate npu-translate.cpp)

llvm_update_compile_flags(npu-translate)
target_link_libraries(npu-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(npu-translate)
