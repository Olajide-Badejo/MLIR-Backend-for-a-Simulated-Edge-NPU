# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUSimulator: the machine, its kernels and its cost model.
#
# It links NPUEncoding and LLVM's Support library and **nothing else**, and in
# particular not MLIR. That is the same split lib/Encoding made and it is here
# for the same payoff: npu-sim reads a .nbin and runs it, and a tool that reads
# a file and executes it has no business linking a compiler. It is also what
# keeps NPUSimulatorTests a binary that compiles and links in seconds, which
# matters for a suite somebody will want to run on every edit.
#
# -Werror=switch, as in lib/Encoding, and for the same reason. Kernels.cpp
# carries a hand written switch over the generated `Opcode` enum with no
# `default` label, which is what assigns every opcode a port. The dispatch table
# beside it covers the kernels themselves through the generated .def. Between
# them, appending an opcode to the ISA description and nothing else stops this
# library compiling until the new opcode has both a port and a kernel.

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Werror=switch)
endif()

# ---------------------------------------------------------------------------
# OpenMP, optional, with a configure time status line.
#
# Section 10.3 requires it to be optional and requires the absence to be said
# out loud rather than to become a silent difference in behaviour. Without it
# the convolution kernel runs single threaded and produces bitwise identical
# output, which is the property SimulatorTest asserts by running the same
# program at one thread and at the maximum and comparing the buffers.
# ---------------------------------------------------------------------------
find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  message(STATUS
    "OpenMP: found ${OpenMP_CXX_VERSION}. The convolution kernel parallelises "
    "over the batch and output channel dimensions; the reductions stay "
    "sequential and in their original order.")
else()
  message(STATUS
    "OpenMP: not found. The convolution kernel runs single threaded. Results "
    "are unchanged: the reduction order is what determines the last bits and "
    "it does not depend on the thread count.")
endif()

add_mlir_library(NPUSimulator
        Simulator.cpp
        Kernels.cpp
        CostModel.cpp
        Memory.cpp

        DEPENDS
        NPUEncodingIncGen

        LINK_LIBS PUBLIC
        NPUEncoding

        LINK_COMPONENTS
        Support
        )

if(OpenMP_CXX_FOUND)
  target_link_libraries(NPUSimulator PUBLIC OpenMP::OpenMP_CXX)
endif()
