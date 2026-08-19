# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUInterfaceTests, the binary the activation table of Section 19.0 switches on
# at P2. It holds the npuisa destination passing and memory effect tests and the
# overlap arithmetic the asynchronous rule is decided on.
#
# It is a different binary from NPUTilingTests on purpose. That one holds the npu
# level tiling interface tests, which arrived a phase earlier, and giving the two
# the same name would mean the P2 guard silently ran P1's tests and reported that
# as P2's gate being met.
#
# NPU_GTEST_LIBRARIES and NPU_GTEST_INCLUDE_DIRS come from the top level
# CMakeLists, which finds either the LLVM build tree's bundled gtest or a system
# GoogleTest package. Naming the variables rather than `llvm_gtest` directly is
# what lets this binary build in CI, where the LLVM prefix is an install rather
# than a build tree and the bundled targets do not exist.

add_llvm_executable(NPUInterfaceTests
        InterfaceTest.cpp
        )

llvm_update_compile_flags(NPUInterfaceTests)

target_include_directories(NPUInterfaceTests SYSTEM PRIVATE
        ${NPU_GTEST_INCLUDE_DIRS}
        )

# MLIRMemRefDialect and MLIRArithDialect are here because the overlap tests parse
# memref.alloc, memref.view and arith.constant: the analysis under test walks
# exactly those operations, so a test of it has to be able to build them.
target_link_libraries(NPUInterfaceTests
        PRIVATE
        MLIRNPU
        MLIRNPUISA
        MLIRArithDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRParser
        MLIRIR
        MLIRSupport
        ${NPU_GTEST_LIBRARIES}
        )
