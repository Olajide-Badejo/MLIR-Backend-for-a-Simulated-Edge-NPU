//===- npu-opt.cpp - NPU optimizer driver ---------------------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// Command line driver for running passes and transforms over NPU dialect IR,
// modeled on mlir-opt. It registers the upstream dialects and passes plus the
// project's own dialects. The npuisa dialect and the project passes are added
// here as they land in later phases.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/Transforms/Passes.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::npu::NPUDialect>();
  registry.insert<mlir::npuisa::NPUISADialect>();
  mlir::npu::registerNPUPasses();
  mlir::npuisa::registerNPUISAPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NPU optimizer driver\n", registry));
}
