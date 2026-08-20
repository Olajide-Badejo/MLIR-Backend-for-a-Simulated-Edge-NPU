# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Two libraries, and the split is the point rather than tidiness.
#
# NPUEncoding is the format itself: the record types, the encoder and decoder,
# the validator and the disassembler. It depends on LLVM's Support library and
# on nothing else, and in particular not on MLIR. That is what lets the fuzz
# target of Section 17.3 and the malformed input corpus link a small library
# rather than the whole compiler, and it is what will let `npu-sim` at P7 read
# a `.nbin` without linking a dialect it does not use.
#
# MLIRNPUEncoding is the bridge: allocated `npuisa` IR in, a `Program` out. It
# is the only half that knows what an operation is.
#
# -Werror=switch is set for this directory and it is load bearing. Section 9.4
# claims that adding an opcode to the ISA description and nothing else fails to
# build in every layer that has no case for it. The generated tables cover the
# mechanical layers on their own; a hand written switch over the generated
# `Opcode` enum with no `default` label is what makes the layers that need a
# human decision fail rather than silently take a default. Validation.cpp's
# semantic dispatch is one of those, and unittests/Encoding carries the other.

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Werror=switch)
endif()

# PARTIAL_SOURCES_INTENDED on both targets. LLVM's build system enforces one
# target per directory and rejects a second one without it. The second target
# is intended and is the split described above: a format library with no MLIR
# dependency, and the bridge that has one.
add_mlir_library(NPUEncoding
        PARTIAL_SOURCES_INTENDED
        Program.cpp
        Validation.cpp
        Disassembler.cpp

        DEPENDS
        NPUEncodingIncGen

        LINK_COMPONENTS
        Support
        )

add_mlir_library(MLIRNPUEncoding
        PARTIAL_SOURCES_INTENDED
        InstructionEncoder.cpp

        DEPENDS
        NPUEncodingIncGen

        LINK_LIBS PUBLIC
        NPUEncoding
        MLIRNPU
        MLIRNPUISA
        MLIRArithDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRIR
        MLIRSupport
        )
