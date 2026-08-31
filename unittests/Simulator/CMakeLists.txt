# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# NPUSimulatorTests, the binary the activation table of Section 19.0 switches on
# at P7. Five files, one binary, and each file is a different kind of test:
#
#   SimulatorTest.cpp       a semantics test per opcode against hand computed
#                           values, covering every case in Section 10.1's P7
#                           list and no integer case
#   CostModelTest.cpp       Section 5.5: the frozen constants, the occupancy
#                           charges, the two timelines, the overlap fraction at
#                           both endpoints, and the single port flag
#   TrapTest.cpp            Section 9.3: the bounds checked accessors, proven
#                           in an assertions build and an NDEBUG build alike
#   DeterminismTest.cpp     Section 10.3: one thread and the maximum thread
#                           count produce bitwise equal buffers
#   DifferentialExport.cpp  the cases the reference interpreter of Section
#                           17.3a is compared against, written out on request
#
# It links NPUSimulator and gtest and nothing else. The simulator carries no
# MLIR dependency, so this binary compiles and links in seconds, which matters
# for a suite somebody will want to run on every edit.

add_llvm_executable(NPUSimulatorTests
        SimulatorTest.cpp
        CostModelTest.cpp
        TrapTest.cpp
        DeterminismTest.cpp
        DifferentialExport.cpp
        )

llvm_update_compile_flags(NPUSimulatorTests)

target_include_directories(NPUSimulatorTests SYSTEM PRIVATE
        ${NPU_GTEST_INCLUDE_DIRS}
        )

target_link_libraries(NPUSimulatorTests
        PRIVATE
        NPUSimulator
        ${NPU_GTEST_LIBRARIES}
        )
