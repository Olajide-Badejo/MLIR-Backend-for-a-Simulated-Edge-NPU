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

# PARTIAL_SOURCES_INTENDED on both targets in this directory. LLVM's build
# system enforces one target per directory and rejects a second one without it.
# The second target is intended and is not a convenience: Section 6's layout
# puts InterfaceTest.cpp and AllocatorTest.cpp side by side here, and Section
# 19.0's activation table switches NPUInterfaceTests on at P2 and
# NPUAllocatorTests at P5. One binary carrying both would mean the P2 guard
# silently running P5's tests, which is the failure mode that table exists to
# prevent.

add_llvm_executable(NPUInterfaceTests
        PARTIAL_SOURCES_INTENDED
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

# NPUAllocatorTests, the binary the activation table switches on at P5. It holds
# Section 17.2's property test: the sweep line peak against a brute force
# recomputation on randomized interval sets, plus the tie breaks of the two
# spill heuristics and the offset arithmetic.
#
# It links MLIRNPUISATransforms rather than MLIRNPUISA, because what it tests
# lives in ScratchpadAllocation.cpp on the transforms side. It needs no dialect
# and no parser: the whole point of keeping the allocator's arithmetic free of
# IR is that a thousand interval sets can be generated as integers rather than
# as modules.

add_llvm_executable(NPUAllocatorTests
        PARTIAL_SOURCES_INTENDED
        AllocatorTest.cpp
        )

llvm_update_compile_flags(NPUAllocatorTests)

target_include_directories(NPUAllocatorTests SYSTEM PRIVATE
        ${NPU_GTEST_INCLUDE_DIRS}
        )

target_link_libraries(NPUAllocatorTests
        PRIVATE
        MLIRNPUISATransforms
        MLIRSupport
        ${NPU_GTEST_LIBRARIES}
        )
