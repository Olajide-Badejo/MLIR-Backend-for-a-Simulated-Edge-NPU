# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# npu-sim links the simulator, the format library and LLVM's Support library,
# and nothing else. That is the payoff of keeping both NPUEncoding and
# NPUSimulator free of MLIR: a tool that reads a file and executes it has no
# business linking a compiler.

set(LLVM_LINK_COMPONENTS
        Support
        )

add_llvm_executable(npu-sim npu-sim.cpp)

llvm_update_compile_flags(npu-sim)
target_link_libraries(npu-sim PRIVATE NPUSimulator)
