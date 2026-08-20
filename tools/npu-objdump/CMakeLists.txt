# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# npu-objdump links the format library and LLVM's Support library, and nothing
# else. That is the payoff of keeping NPUEncoding free of MLIR: a tool that
# reads a file and prints it has no business linking a compiler.

set(LLVM_LINK_COMPONENTS
        Support
        )

add_llvm_executable(npu-objdump npu-objdump.cpp)

llvm_update_compile_flags(npu-objdump)
target_link_libraries(npu-objdump PRIVATE NPUEncoding)
