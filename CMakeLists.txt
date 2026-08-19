# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The structure of this file follows mlir/examples/standalone/CMakeLists.txt
# from the LLVM project, which is the supported way to build an out of tree
# MLIR dialect. The parts that are recognisably that example are marked below.

cmake_minimum_required(VERSION 3.20.0)
project(npu-mlir LANGUAGES CXX C)

# LLVM 22 builds as C++17. Building this project as C++20 against a C++17 LLVM
# is fine in one direction only: our headers may use C++20, LLVM's may not
# require it. Nothing here relies on a C++20 feature crossing the ABI boundary.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- begin: derived from the LLVM standalone example ------------------------
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
# --- end: derived from the LLVM standalone example --------------------------

# `lit -sv` is what the standalone example defaults to and what every phase
# gate in the build specification quotes output from. --allow-empty-runs is on
# top of that and is load bearing exactly once: at P0 the suite is empty by
# design, because the dialect that the tests test does not exist until P1, and
# without the flag lit exits 2 on "did not discover any tests" and the P0 gate
# cannot pass. It stays after P1 so that running a single subdirectory suite
# that happens to be empty is a pass rather than an error.
set(LLVM_LIT_ARGS "-sv --allow-empty-runs" CACHE STRING "Default options for lit")

set(NPU_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(NPU_BINARY_DIR ${PROJECT_BINARY_DIR})

# The MLIR Python bindings live in the LLVM build tree and are not pip
# installed, so anything that imports `mlir` needs this on PYTHONPATH. Section
# 3.3 of the build specification requires the same path in three places:
# pyproject.toml, test/lit.cfg.py and the pytest configuration. Deriving it
# from MLIR_DIR rather than hardcoding $HOME means a relocated LLVM build
# fails at configure time with a readable message instead of failing at import
# time inside a test that has nothing to do with it.
set(MLIR_PYTHON_PACKAGES_DIR "${LLVM_BINARY_DIR}/tools/mlir/python_packages/mlir_core"
    CACHE PATH "Directory holding the MLIR Python bindings package")
# The check looks for mlir/ir.py rather than mlir/__init__.py: the bindings
# package is a namespace package and ships no __init__.py at all, so checking
# for one reports a working install as missing.
if(NOT EXISTS "${MLIR_PYTHON_PACKAGES_DIR}/mlir/ir.py")
  message(WARNING
    "MLIR Python bindings not found under ${MLIR_PYTHON_PACKAGES_DIR}. "
    "Tests that import `mlir` will fail. Set MLIR_PYTHON_PACKAGES_DIR to the "
    "python_packages/mlir_core directory of the LLVM build.")
endif()
message(STATUS "MLIR Python packages dir: ${MLIR_PYTHON_PACKAGES_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${NPU_SOURCE_DIR}/include)
include_directories(${NPU_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

# ---------------------------------------------------------------------------
# GoogleTest, out of tree.
#
# LLVM's own `add_unittest` is not available to an out of tree project: it lives
# inside the LLVM build and is not exported. What is available is the bundled
# gtest that an LLVM *build tree* carries, as two static libraries and a pair of
# header directories. An installed LLVM *prefix* carries neither, which is the
# case in CI, so this whole section is guarded and says in the configure log
# which of the two it found.
#
# A phase that runs its unit tests locally while CI cannot build them is not a
# comfortable state, but it is the honest one at P1 and it is better stated here
# than discovered as a missing target. The activation table already keeps every
# GoogleTest binary guarded off in CI until its own phase.
#
# The `llvm_gtest` and `llvm_gtest_main` targets themselves come from
# LLVMConfig.cmake, which an LLVM build tree exports and an install prefix does
# not, so the target existing is the guard. The headers are not on the exported
# target's interface, so they are added by path.
# ---------------------------------------------------------------------------
set(NPU_LLVM_THIRD_PARTY "${LLVM_BUILD_MAIN_SRC_DIR}/../third-party"
    CACHE PATH "LLVM's third-party directory, which carries the bundled gtest")

if(TARGET llvm_gtest
   AND EXISTS "${NPU_LLVM_THIRD_PARTY}/unittest/googletest/include/gtest/gtest.h")
  message(STATUS "LLVM gtest found: building the unit test binaries")
  set(NPU_LLVM_GTEST_INCLUDE_DIRS
      "${NPU_LLVM_THIRD_PARTY}/unittest/googletest/include"
      "${NPU_LLVM_THIRD_PARTY}/unittest/googlemock/include")
  add_subdirectory(unittests)
else()
  message(STATUS
    "LLVM gtest not found: the unit test binaries are not built. This is "
    "expected against an installed LLVM prefix, which exports neither the "
    "llvm_gtest target nor the bundled gtest headers.")
endif()

# ---------------------------------------------------------------------------
# npu-dialect-doc
#
# Regenerates docs/DIALECT_REFERENCE.md from the ODS source, so the reference
# cannot drift from the dialect it documents. The generated file is committed
# and CI regenerates it and diffs, which is what makes the drift a build
# failure rather than a thing somebody notices months later.
#
# The generation itself is add_mlir_doc in include/NPU/Dialect/NPU/IR. This
# target is the copy step plus the header that says the file is generated, and
# it is separate so that the committed file carries that header while the raw
# mlir-tblgen output does not have to.
# ---------------------------------------------------------------------------
add_custom_target(npu-dialect-doc
  COMMAND ${CMAKE_COMMAND}
          -DGENERATED=${CMAKE_BINARY_DIR}/docs/NPU/NPUDialectReference.md
          -DOUTPUT=${NPU_SOURCE_DIR}/docs/DIALECT_REFERENCE.md
          -P ${NPU_SOURCE_DIR}/cmake/WriteDialectReference.cmake
  DEPENDS NPUDialectReferenceDocGen
  COMMENT "Regenerating docs/DIALECT_REFERENCE.md from the ODS source"
  VERBATIM)
