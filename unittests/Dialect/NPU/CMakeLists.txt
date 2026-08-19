# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUTilingTests, not NPUInterfaceTests. The activation table names
# NPUInterfaceTests as the binary that switches on at P2, and that binary holds
# the npuisa destination passing and memory effect tests. This one holds the
# npu level tiling interface tests, which are a different thing arriving a phase
# earlier, and giving them the same name would mean the P2 guard silently
# started running P1's tests and reporting that as P2's gate being met.
#
# add_llvm_executable rather than add_mlir_unittest. The latter is a thin
# wrapper over LLVM's add_unittest, which exists only inside an LLVM build and
# is not exported to an out of tree project, so it is simply not a command here.
# What an out of tree project has instead is whichever GoogleTest the top level
# CMakeLists found, the LLVM build tree's bundled one or a system package, named
# by NPU_GTEST_LIBRARIES and NPU_GTEST_INCLUDE_DIRS so that this file does not
# have to know which. That is a real difference between building in tree and out
# of tree, and it is written here rather than being rediscovered by whoever adds
# the next test binary.
#
# The binary is built and run locally at P1. CI does not run it: the activation
# table switches GoogleTest binaries on by name and NPUTilingTests has no row,
# because adding one is a change to the table rather than something a phase does
# on its own.

add_llvm_executable(NPUTilingTests
        TilingInterfaceTest.cpp
        )

llvm_update_compile_flags(NPUTilingTests)

target_include_directories(NPUTilingTests SYSTEM PRIVATE
        ${NPU_GTEST_INCLUDE_DIRS}
        )

target_link_libraries(NPUTilingTests
        PRIVATE
        MLIRNPU
        MLIRNPUTilingInterfaceImpl
        MLIRFuncDialect
        MLIRTensorDialect
        MLIRParser
        MLIRIR
        MLIRSupport
        ${NPU_GTEST_LIBRARIES}
        )
