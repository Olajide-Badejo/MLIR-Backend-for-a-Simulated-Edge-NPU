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
// **`--npu-pass-stats-json` needs the `PassManager` that `MlirOptMain` builds,
// and that is why this file stopped being a one line `main`.** Section 16.2's
// instrumentation has to be attached to the pass manager the pipeline actually
// runs on, and `MlirOptMain(argc, argv, toolName, registry)` builds that manager
// inside itself and hands it out through exactly one hook:
// `MlirOptMainConfig::setPassPipelineSetupFn`, the callback it calls with the
// real manager just before it runs. So the four argument overload is unrolled
// below into what it does, with the instrumentation installed inside a callback
// that then chains to the one the command line already set up.
//
// **The default path is unchanged and is still the library's own overload.**
// The unrolled path runs only when `--npu-pass-stats-json` is given. That is
// deliberate: the unrolled form does not reproduce `--show-dialects` and
// `--list-passes`, which are answered inside the library by functions an out of
// tree tool cannot call, and a driver that quietly lost two flags in order to
// gain one would be a bad trade. Every invocation this project makes without the
// statistics flag takes exactly the path it took before.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/Interfaces/NPUTilingInterfaceImpl.h"
#include "NPU/Dialect/NPU/Transforms/Passes.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/Transforms/Passes.h"
#include "NPU/Pipeline/PassStats.h"
#include "NPU/Pipeline/Pipeline.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

constexpr llvm::StringLiteral kDescribeFlag = "--npu-describe-pipeline";

llvm::cl::opt<bool> describePipeline(
    kDescribeFlag.drop_front(2),
    llvm::cl::desc("Print the -O level table as JSON and exit. Answers a "
                   "question about the compiler rather than about a file, so "
                   "it takes no input."),
    llvm::cl::init(false));

llvm::cl::opt<std::string> passStatsJson(
    "npu-pass-stats-json",
    llvm::cl::desc("Write the Section 16.2 per pass operation counts and wall "
                   "clock as JSON to this path. The counts are computed by a "
                   "PassInstrumentation on the pass manager this tool runs, "
                   "not by any flag. Cross check the wall clock against "
                   "--mlir-timing, which measures the same run."),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

/// The four argument `MlirOptMain`, unrolled far enough to reach the config.
///
/// This is `mlir::MlirOptMain(argc, argv, inputFilename, outputFilename,
/// registry)` with one addition: the pipeline setup callback is wrapped so that
/// the instrumentation is installed on the real manager before the command
/// line's own pipeline is added to it. Nothing else about the sequence changes.
int runWithPassStatistics(int argc, char **argv,
                          llvm::StringRef inputFilename,
                          llvm::StringRef outputFilename,
                          mlir::DialectRegistry &registry,
                          llvm::StringRef statsPath) {
  llvm::InitLLVM lifetime(argc, argv);

  mlir::MlirOptMainConfig fromCommandLine =
      mlir::MlirOptMainConfig::createFromCLOptions();

  // A copy, so that the wrapper below can call the original callback rather
  // than replace it. The copy is safe because the callback the command line
  // installed captures two things that outlive this function: the process wide
  // pipeline parser and the process wide option storage.
  mlir::MlirOptMainConfig config = fromCommandLine;
  config.setPassPipelineSetupFn(
      [fromCommandLine, path = statsPath.str()](mlir::PassManager &pm) {
        mlir::npu::pipeline::installPassStatistics(pm, path);
        return fromCommandLine.setupPassPipeline(pm);
      });

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(mlir::MlirOptMain(output->os(), std::move(file), registry, config)))
    return 1;

  output->keep();
  return 0;
}

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

  auto [inputFilename, outputFilename] = mlir::registerAndParseCLIOptions(
      argc, argv, "NPU optimizer driver\n", registry);

  if (passStatsJson.empty())
    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, inputFilename, outputFilename, registry));

  return runWithPassStatistics(argc, argv, inputFilename, outputFilename,
                               registry, passStatsJson);
}
