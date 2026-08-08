# npu-mlir: out of tree MLIR compiler backend for a simulated edge NPU.
#
# This project links against a prebuilt LLVM/MLIR (built once at the pinned tag,
# see docs/BUILD.md). It never builds LLVM itself. Point the configure step at
# the MLIR install or build tree with -DMLIR_DIR=/path/to/lib/cmake/mlir.
cmake_minimum_required(VERSION 3.24)

project(npu-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Custom dialects, passes, libraries, and tools are added phase by phase.
# TableGen include directories are descended into first so their generated
# headers exist before the libraries and tools that consume them are built.
add_subdirectory(include/NPU/Dialect/NPU/IR)
add_subdirectory(include/NPU/Dialect/NPU/Transforms)
add_subdirectory(include/NPU/Dialect/NPUISA/IR)
add_subdirectory(include/NPU/Dialect/NPUISA/Transforms)
add_subdirectory(lib/Dialect/NPU/IR)
add_subdirectory(lib/Dialect/NPU/Transforms)
add_subdirectory(lib/Dialect/NPUISA/IR)
add_subdirectory(lib/Dialect/NPUISA/Transforms)
add_subdirectory(lib/Encoding)
add_subdirectory(lib/Simulator)
add_subdirectory(tools/npu-opt)
add_subdirectory(tools/npu-objdump)
add_subdirectory(tools/npu-translate)
add_subdirectory(tools/npu-sim)
add_subdirectory(unittests)
add_subdirectory(test)

# Re-measure the recorded regression baseline and fail on any drift. The script
# skips its own build step here, since ninja has just done it, and depends on
# the tools and test binaries the measurement needs.
add_custom_target(baseline-check
  # Invoked through bash rather than directly: this repository has
  # core.fileMode=false, so the executable bit is not reliably carried.
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/regression-baseline.sh
          --check --no-build --build-dir ${CMAKE_BINARY_DIR}
  DEPENDS npu-opt npu-translate npu-objdump npu-sim
          NPUEncodingTests NPUSimulatorTests
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  USES_TERMINAL
  COMMENT "Checking the recorded regression baseline"
)
set_target_properties(baseline-check PROPERTIES FOLDER "Tests")
