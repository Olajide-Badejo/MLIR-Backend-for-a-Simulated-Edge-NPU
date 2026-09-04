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
# output, which is the property DeterminismTest asserts by running the same
# program at one thread and at the maximum and comparing the buffers.
#
# **The compile option is set at directory scope and that is D-0047.** It is
# not a style choice and it is not interchangeable with attaching the imported
# target to `NPUSimulator` below. `add_mlir_library` compiles these four sources
# in an object library called `obj.NPUSimulator` and then builds `NPUSimulator`
# out of the objects, so a usage requirement attached to `NPUSimulator` reaches
# everything that links it and does not reach the sources it is made of. From
# P7 until P12 the only OpenMP wiring here was the `target_link_libraries` line
# at the bottom of this file, and the result was that `-fopenmp` reached
# `unittests/Simulator/DeterminismTest.cpp`, which links `NPUSimulator`, and
# never reached `Kernels.cpp`, which is `NPUSimulator`. The `#ifdef _OPENMP`
# around the convolution's `parallel for` was compiled out in every build in
# every environment, the status line above said the kernel parallelises, and the
# test that exists to prove the parallel kernel is bitwise stable was comparing
# one serial run against another. `add_compile_options` is directory scope, so
# it applies to every target created after it in this file, `obj.NPUSimulator`
# included. It is the same mechanism `-Werror=switch` above already relies on
# for the same reason.
#
# `separate_arguments` because `OpenMP_CXX_FLAGS` is one string and is more than
# one flag on some compilers, and `add_compile_options` would pass a multi word
# string to the compiler as a single argument.
#
# The link line at the bottom of the file stays: it is what puts the runtime on
# the link line of everything that links this library. The two are different
# jobs and this project needed both all along.
# ---------------------------------------------------------------------------
find_package(OpenMP COMPONENTS CXX)

if(OpenMP_CXX_FOUND)
  message(STATUS
    "OpenMP: found ${OpenMP_CXX_VERSION}. The convolution kernel parallelises "
    "over the batch and output channel dimensions; the reductions stay "
    "sequential and in their original order.")
  separate_arguments(NPU_OPENMP_CXX_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
  add_compile_options(${NPU_OPENMP_CXX_FLAGS})
else()
  message(STATUS
    "OpenMP: not found. The convolution kernel runs single threaded. Results "
    "are unchanged: the reduction order is what determines the last bits and "
    "it does not depend on the thread count.")
endif()

# `CostModel.cpp` used to be a source of this library and moved to
# `lib/CostModel` at P13, when `-npu-tile-to-scratchpad` became a second caller
# of the same charges. `NPUCostModel` links nothing, so depending on it here
# costs this library nothing, and it is what stops a compiler pass having to
# link a simulator to reach the arithmetic Section 5.5 says they must share.
add_mlir_library(NPUSimulator
        Simulator.cpp
        Kernels.cpp
        Memory.cpp

        DEPENDS
        NPUEncodingIncGen

        LINK_LIBS PUBLIC
        NPUEncoding
        NPUCostModel

        LINK_COMPONENTS
        Support
        )

if(OpenMP_CXX_FOUND)
  target_link_libraries(NPUSimulator PUBLIC OpenMP::OpenMP_CXX)
endif()
