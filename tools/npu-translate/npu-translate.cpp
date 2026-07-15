//===- npu-translate.cpp - Encode allocated npuisa MLIR to .nbin ----------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Encoding/InstructionEncoder.h"
#include "NPU/Encoding/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::string input, output;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc)
      output = argv[++i];
    else
      input = arg;
  }
  if (input.empty() || output.empty()) {
    std::cerr << "usage: npu-translate <input.mlir> -o <output.nbin>\n";
    return 1;
  }

  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                      mlir::npu::NPUDialect, mlir::npuisa::NPUISADialect>();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module) {
    std::cerr << "npu-translate: failed to parse " << input << "\n";
    return 1;
  }

  mlir::func::FuncOp func;
  module->walk([&](mlir::func::FuncOp f) {
    if (!func)
      func = f;
  });
  if (!func) {
    std::cerr << "npu-translate: no function found\n";
    return 1;
  }

  auto program = npu::encodeFunction(func);
  if (mlir::failed(program))
    return 1;

  std::vector<uint8_t> bytes = program->encode();
  std::ofstream out(output, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return 0;
}
