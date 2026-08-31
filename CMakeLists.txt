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
# ---------------------------------------------------------------------------
# NDEBUG, which this project cannot otherwise have.
#
# Section 9.3 requires the simulator's bounds checked accessors to refuse
# gracefully in **every build mode** and requires a test to prove it in a
# release build and an assertions build alike. Producing the first of those is
# not a matter of `-DCMAKE_BUILD_TYPE=Release`, and that is worth stating here
# because the obvious attempt looks like it worked. `LLVMConfig.cmake` sets
# `LLVM_ENABLE_ASSERTIONS` as a plain variable from the LLVM tree this project
# is configured against, which shadows any cache value the command line offers.
# `HandleLLVMOptions` then reads it and adds `_DEBUG`, `_GLIBCXX_ASSERTIONS`
# and an explicit `-UNDEBUG`, and that `-UNDEBUG` lands **after** the `-DNDEBUG`
# a Release configuration supplies. So a Release build against an assertions
# LLVM keeps its assertions and looks like a release build from the outside.
# That is D-0028: the first NDEBUG run of the P7 gate was made against exactly
# such a build, and it passed, and it proved nothing.
#
# This block therefore sits **before** `include(HandleLLVMOptions)`, which is
# where those flags are decided. Putting it after, which was the first attempt,
# changes a variable nothing reads again. Setting it here is also what clears
# `LLVM_DEFINITIONS`, because `HandleLLVMOptions` rebuilds that variable from
# the same switch; the second half of this option, at the `add_definitions` call
# below, is only about how the cleared string is then parsed. The LLVM at
# `~/llvm-project/build` is an assertions build and is not this project's to
# rebuild, so the switch lives here rather than in the toolchain, and it moves
# this project's own targets only.
#
# `docs/BUILD.md` carries the configure line and `docs/PHASE_STATE.md` records
# the four runs it exists to make possible.
# ---------------------------------------------------------------------------
option(NPU_FORCE_NDEBUG
       "Compile this project's targets with NDEBUG and without libstdc++ assertions, even when the LLVM they are built against is an assertions build."
       OFF)

if(NPU_FORCE_NDEBUG)
  set(LLVM_ENABLE_ASSERTIONS OFF)
  message(STATUS
    "NDEBUG: forced on. NDEBUG is defined, _DEBUG and _GLIBCXX_ASSERTIONS are "
    "not, and no -UNDEBUG is appended. This is the build mode Section 9.3's "
    "'every build mode' clause names.")
else()
  message(STATUS
    "NDEBUG: not forced. This build inherits the assertion setting of the LLVM "
    "tree it is configured against, which for an assertions LLVM means "
    "assertions stay on whatever CMAKE_BUILD_TYPE says. Configure a second "
    "directory with -DNPU_FORCE_NDEBUG=ON for the release half of Section "
    "9.3's contract.")
endif()

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

# `LLVM_DEFINITIONS` is one space separated string rather than a list, and
# `add_definitions` splits it on whitespace only as far as its own parse gets:
# with `-D_DEBUG` leading it splits into six clean definitions, and with
# `-D_GLIBCXX_USE_CXX11_ABI=1` leading, which is what is left once
# `NPU_FORCE_NDEBUG` has taken the assertion macros out, it stops at the `=` and
# hands the compiler one definition whose value is the rest of the line. That is
# twelve `_GLIBCXX_USE_CXX11_ABI redefined` warnings for a build that is
# otherwise correct, and it is why the forced path splits the string itself
# rather than letting the command guess.
#
# The unforced path is left exactly as it was. It works, this project has built
# through six phases on it, and changing a line that is not wrong to make two
# branches look alike is how a working build acquires a new failure.
if(NPU_FORCE_NDEBUG)
  separate_arguments(NPU_LLVM_DEFINITION_LIST NATIVE_COMMAND
                     "${LLVM_DEFINITIONS}")
  add_definitions(${NPU_LLVM_DEFINITION_LIST})
  # Explicit rather than left to CMAKE_CXX_FLAGS_<CONFIG>, so that the option
  # means what its name says under a Debug configuration too.
  add_compile_definitions(NDEBUG)
  if("${LLVM_DEFINITIONS}" MATCHES "_GLIBCXX_ASSERTIONS|-D_DEBUG|-UNDEBUG")
    message(FATAL_ERROR
      "NPU_FORCE_NDEBUG cannot hold: LLVM_DEFINITIONS still carries an "
      "assertion macro (${LLVM_DEFINITIONS}), so this build would define "
      "NDEBUG and keep its assertions, which is the thing D-0028 was. Refusing "
      "rather than producing it.")
  endif()
else()
  add_definitions(${LLVM_DEFINITIONS})
endif()

# ---------------------------------------------------------------------------
# The ISA generator, added before include/ rather than with the other tools.
#
# Section 9.4's generation runs at build time from include/NPU/Encoding, and a
# custom command there names $<TARGET_FILE:npu-isa-tblgen>. Generator
# expressions resolve at generate time rather than at the point of the call, so
# the ordering is not strictly required by CMake. It is here because a reader
# who finds a target used before it is defined has to go and check that, and
# the cost of sparing them that is one line in the right place.
# ---------------------------------------------------------------------------
add_subdirectory(tools/npu-isa-tblgen)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)

# ---------------------------------------------------------------------------
# Fuzzing, off by default.
#
# Section 17.3's coverage guided target needs -fsanitize=fuzzer, which is a
# clang feature. The default build here is gcc against a gcc built LLVM, so the
# target sits behind an option rather than behind a compiler check: a compiler
# check would silently drop the target in the one build where somebody wanted
# it and report nothing, and this project's rule is that a step which is off
# says so. The sanitizers job of Section 19 configures a separate build
# directory with clang and turns this on.
# ---------------------------------------------------------------------------
option(NPU_ENABLE_FUZZERS
       "Build the coverage guided fuzz targets. Needs a clang that supports -fsanitize=fuzzer."
       OFF)

if(NPU_ENABLE_FUZZERS)
  message(STATUS "Fuzzers: ON. fuzz/ is built with -fsanitize=fuzzer.")
  add_subdirectory(fuzz)
else()
  message(STATUS
    "Fuzzers: OFF. Configure with -DNPU_ENABLE_FUZZERS=ON and a clang "
    "toolchain to build fuzz/nbin_decode_fuzzer.")
endif()

# ---------------------------------------------------------------------------
# GoogleTest, out of tree, from either of two sources.
#
# LLVM's own `add_unittest` is not available to an out of tree project: it lives
# inside the LLVM build and is not exported. So this project finds a GoogleTest
# for itself, and there are two places one can come from, which is why this is
# a search with a fallback rather than a single guarded branch.
#
# 1. **An LLVM build tree** carries a bundled gtest as two static libraries,
#    `llvm_gtest` and `llvm_gtest_main`, exported by LLVMConfig.cmake, plus a
#    pair of header directories under `third-party` that are *not* on the
#    exported target's interface and so are added by path. This is what a local
#    developer checkout has and it is preferred when present, because it is the
#    same gtest LLVM itself was built and tested against, so there is no chance
#    of an ABI or flag mismatch between the two.
#
# 2. **A system GoogleTest package**, found through `find_package(GTest)`. This
#    is what an image built around an installed LLVM *prefix* has, and CI is
#    exactly that: an install prefix exports neither the `llvm_gtest` target nor
#    the bundled headers, so before this fallback existed the unit test binaries
#    simply were not built in CI and the P2 activation of `NPUInterfaceTests`
#    would have had nothing to run. `docker/Dockerfile.llvm` installs
#    `googletest` into the final image for this branch to find.
#
# Both paths end at the same two variables, `NPU_GTEST_LIBRARIES` and
# `NPU_GTEST_INCLUDE_DIRS`, so no test CMakeLists below has to know which one
# was taken. The configure log says which, because "the tests were not built"
# and "the tests passed" must never look the same from the outside.
# ---------------------------------------------------------------------------
set(NPU_LLVM_THIRD_PARTY "${LLVM_BUILD_MAIN_SRC_DIR}/../third-party"
    CACHE PATH "LLVM's third-party directory, which carries the bundled gtest")

set(NPU_GTEST_FOUND OFF)

if(TARGET llvm_gtest
   AND EXISTS "${NPU_LLVM_THIRD_PARTY}/unittest/googletest/include/gtest/gtest.h")
  message(STATUS
    "GoogleTest: using the LLVM build tree's bundled gtest "
    "(llvm_gtest, llvm_gtest_main). Building the unit test binaries.")
  set(NPU_GTEST_LIBRARIES llvm_gtest llvm_gtest_main)
  set(NPU_GTEST_INCLUDE_DIRS
      "${NPU_LLVM_THIRD_PARTY}/unittest/googletest/include"
      "${NPU_LLVM_THIRD_PARTY}/unittest/googlemock/include")
  set(NPU_GTEST_FOUND ON)
else()
  # CONFIG mode is not required: a distribution's googletest ships a CMake
  # package config on some systems and only FindGTest's module on others, and
  # the default search tries both. GTest::gtest_main pulls GTest::gtest with it
  # through its own interface, so linking the main library alone would be
  # enough; both are named for symmetry with the branch above and because an
  # older FindGTest that defines only the plain variables still populates
  # GTEST_BOTH_LIBRARIES, which the imported targets cover.
  find_package(GTest QUIET)
  if(GTest_FOUND OR GTEST_FOUND)
    message(STATUS
      "GoogleTest: using the system GoogleTest package. Building the unit "
      "test binaries.")
    set(NPU_GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
    set(NPU_GTEST_INCLUDE_DIRS "")
    set(NPU_GTEST_FOUND ON)
  endif()
endif()

if(NPU_GTEST_FOUND)
  add_subdirectory(unittests)
else()
  message(STATUS
    "GoogleTest: not found in the LLVM build tree and no system package "
    "either, so the unit test binaries are not built. Install googletest, or "
    "point NPU_LLVM_THIRD_PARTY at an LLVM build tree's third-party directory.")
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

# ---------------------------------------------------------------------------
# npu-isa-doc
#
# The same mechanism one level down, for the instruction set. Section 9.4 asks
# for the ISA staleness gate to be identical in shape to the dialect
# reference's so that there is one staleness mechanism in this project rather
# than two, and this is that: generate, copy over the committed file, and let
# CI regenerate and diff.
#
# Two files are copied rather than one. docs/ISA_MANUAL.md is the manual, and
# only its marked sections are replaced, because the rest of it is hand written
# prose that no generator could produce. docs/ISA_OPCODES.json is the machine
# readable opcode list scripts/check-reachability.py reads, which is Section
# 9.4's last generated layer: the reachability checker's opcode list, which
# before this phase was parsed back out of source files.
# ---------------------------------------------------------------------------
add_custom_target(npu-isa-doc
  COMMAND ${CMAKE_COMMAND}
          -DGENERATED=${NPU_ISA_GENERATED_MANUAL}
          -DOUTPUT=${NPU_SOURCE_DIR}/docs/ISA_MANUAL.md
          -DJSON_GENERATED=${NPU_ISA_GENERATED_JSON}
          -DJSON_OUTPUT=${NPU_SOURCE_DIR}/docs/ISA_OPCODES.json
          -P ${NPU_SOURCE_DIR}/cmake/WriteISAManual.cmake
  DEPENDS NPUEncodingIncGen
  COMMENT "Regenerating the generated sections of docs/ISA_MANUAL.md"
  VERBATIM)
