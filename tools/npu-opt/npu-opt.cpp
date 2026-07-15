//===- npu-opt.cpp - NPU optimizer driver ---------------------------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// Command line driver for running passes and transforms over NPU dialect IR,
// modeled on mlir-opt. Phase 0 registers the upstream dialects and passes only;
// the npu and npuisa dialects and their passes are registered here as they land
// in later phases.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU optimizer driver\n", registry));
}
