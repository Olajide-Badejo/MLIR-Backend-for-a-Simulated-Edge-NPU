# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUSimulator: the machine's two memories and the arithmetic of its cost model.
#
# It links NPUEncoding and LLVM's Support library and **nothing else**, and in
# particular not MLIR. That is the same split lib/Encoding made and it is here
# for the same payoff: npu-sim reads a .nbin and runs it, and a tool that reads
# a file and executes it has no business linking a compiler. It is also what
# keeps NPUSimulatorTests a binary that compiles and links in seconds, which
# matters for a suite somebody will want to run on every edit.
#
# The kernels and the executor arrive in the next commit and this file grows to
# meet them.

add_mlir_library(NPUSimulator
        CostModel.cpp
        Memory.cpp

        DEPENDS
        NPUEncodingIncGen

        LINK_LIBS PUBLIC
        NPUEncoding

        LINK_COMPONENTS
        Support
        )
