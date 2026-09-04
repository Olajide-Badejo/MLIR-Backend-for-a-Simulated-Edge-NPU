# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUCostModel: Section 5.5's charges, on their own, because two callers need
# them and neither should have to link the other's world to get them.
#
# **Split out of lib/Simulator at P13, and the reason is the one home rule
# rather than tidiness.** `-npu-tile-to-scratchpad` scores its candidate
# mappings with the same `gemmCharge`, `conv2dCharge` and `dmaCycles` the
# simulator charges with, which is what Section 5.5 means when it says the one
# home "is also what the tiling pass scores against, so the project has exactly
# one cost function rather than a modelled one and a heuristic one that can
# disagree". Before this split the only way for a pass to reach those functions
# was to link `NPUSimulator`, which would have put the machine, its kernels, its
# memory and its OpenMP runtime on the link line of `npu-opt` and of every
# dialect library that transitively reached it. A compiler has no business
# linking a simulator, which is the same argument lib/Simulator's own header
# makes in the other direction about a simulator linking MLIR.
#
# It is two functions over the constants in `include/NPU/Simulator/CostModel.h`
# and it links **nothing at all**, which is what makes it safe for either side
# to depend on. The header keeps its path: it is still the simulator's cost
# model, it is just no longer only the simulator's.
#
# It is its own directory rather than a second target in lib/Simulator because
# LLVM's build system enforces one target per directory unless a target opts out
# with PARTIAL_SOURCES_INTENDED, and that opt out is discouraged upstream for a
# reason this project agrees with: a directory whose sources belong to two
# targets is a directory where adding a file does not obviously add it to
# anything.

add_mlir_library(NPUCostModel
        CostModel.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NPU/Simulator
        )
