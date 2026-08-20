# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUEncodingTests, the binary the activation table of Section 19.0 switches on
# at P6. Four files, one binary, and each file is a different kind of test:
#
#   EncodingTest.cpp       hand written round trips, the format's arithmetic,
#                          and the constants that must never move
#   ValidationTest.cpp     every named check of Section 9.2, triggered
#   PropertyTest.cpp       Section 17.2: randomized programs, 1000 iterations
#                          at a fixed seed, structural equality plus byte
#                          identical re-encoding
#   MalformedInputTest.cpp Section 17.3: the seed and regression corpus, over
#                          three hundred deliberately malformed files, every one
#                          rejected cleanly and none allocating more than a few
#                          megabytes
#
# **The corpus is never called a fuzzer**, here or anywhere else. It is a
# negative test table whose power is capped at what its author imagined on the
# day it was written, and naming it after something that grows would be a claim
# it cannot meet. The coverage guided target lives in fuzz/.
#
# It links NPUEncoding and gtest and nothing else. The format library carries no
# MLIR dependency, so this binary compiles and links in seconds, which matters
# for a corpus of three hundred cases that somebody will want to run often.
#
# -Werror=switch, as in lib/Encoding, and for the same reason: PropertyTest's
# per opcode builder is a switch over the generated Opcode enum with no default
# label, so adding an opcode to the ISA description and nothing else stops this
# binary compiling until the property test can build one.

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Werror=switch)
endif()

add_llvm_executable(NPUEncodingTests
        EncodingTest.cpp
        ValidationTest.cpp
        PropertyTest.cpp
        MalformedInputTest.cpp
        )

llvm_update_compile_flags(NPUEncodingTests)

target_include_directories(NPUEncodingTests SYSTEM PRIVATE
        ${NPU_GTEST_INCLUDE_DIRS}
        )

target_link_libraries(NPUEncodingTests
        PRIVATE
        NPUEncoding
        ${NPU_GTEST_LIBRARIES}
        )
