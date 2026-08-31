//===- npu-opt.cpp --------------------------------------------*- C++ -*-===//
//
// This file is derived from mlir/examples/standalone/standalone-opt/
// standalone-opt.cpp in the LLVM project and keeps that file's licence.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-FileCopyrightText: Copyright (c) LLVM Project contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The optimizer driver for this project. It registers the upstream MLIR
// dialects and passes, this project's own dialects, and the external interface
// models those dialects promise.
//
// The tiling interface registration is a separate call rather than something
// the dialect does for itself, and that is the whole point of the promised
// interface mechanism: the dialect library carries no dependency on the tiling
// stack, and a tool that forgets this line gets a named error at the first use
// saying the interface was promised and never provided.
//
// This project's own passes are registered by their generated entry point,
// beside MLIR's. The `-O` levels join them at P8, from `lib/Pipeline`, which is
// the phase that finally has a whole pipeline to describe. Until then no
// pipeline name was registered at all, because a name registered before there
// is anything to put behind it promises more than it delivers.
//
// **`--npu-describe-pipeline` is handled before `MlirOptMain` and that is
// deliberate.** Every other option here is a question about an input file, and
// `MlirOptMain` owns the parsing and requires one. This flag asks a question
// about the compiler instead: which passes each level runs, and which of them
// Section 16.2's ablation may remove. Section 16.2 requires the driver to read
// that set at run time rather than keep a copy, so the answer has to be
// obtainable without a file to run it on. The option is registered with
// `llvm::cl` as well, so that `--help` lists it beside everything else rather
// than leaving it as folklore.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/Interfaces/NPUTilingInterfaceImpl.h"
#include "NPU/Dialect/NPU/Transforms/Passes.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/Transforms/Passes.h"
#include "NPU/Pipeline/Pipeline.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace {

constexpr llvm::StringLiteral kDescribeFlag = "--npu-describe-pipeline";

llvm::cl::opt<bool> describePipeline(
    kDescribeFlag.drop_front(2),
    llvm::cl::desc("Print the -O level table as JSON and exit. Answers a "
                   "question about the compiler rather than about a file, so "
                   "it takes no input."),
    llvm::cl::init(false));

} // namespace

int main(int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    if (kDescribeFlag == argv[index]) {
      mlir::npu::pipeline::printDescriptionAsJson(llvm::outs());
      return 0;
    }
  }

  mlir::registerAllPasses();
  mlir::npu::registerNPUPasses();
  mlir::npuisa::registerNPUISAPasses();
  mlir::npu::pipeline::registerNPUPipelines();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  registry.insert<mlir::npu::NPUDialect, mlir::npuisa::NPUISADialect>();
  mlir::npu::registerNPUTilingInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NPU optimizer driver\n", registry));
}
