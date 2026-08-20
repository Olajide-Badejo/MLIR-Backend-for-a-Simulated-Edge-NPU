//===- npu-translate.cpp - MLIR to .nbin ----------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Allocated `npuisa` IR in, a `.nbin` out.
//
//   npu-opt model.mlir --npu-lower-to-npuisa --npu-allocate-scratchpad \
//     | npu-translate -o model.nbin
//
// Two rules from the Phase P6 roadmap entry shape the control flow here, and
// they are the same rule seen twice: **it fails and writes no output file on
// an operation it cannot encode, and the output file is not created before the
// encode result is known.** So the whole encode happens into memory, then the
// program is validated, and only then is a file opened. A tool that opened the
// file first would leave a zero byte `.nbin` behind on every failure, and a
// build system that treats an existing file as an up to date one would then
// carry that emptiness forward.
//
// The self validation is not defensive decoration. The encoder is the only
// producer of `.nbin` files this project has, so a bug in it is a bug that
// every downstream consumer inherits, and `validate()` is the one place that
// knows all thirty three rules. Catching an encoder bug here costs a
// millisecond and names the check; catching it in the simulator costs a
// debugging session.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Encoding/InstructionEncoder.h"
#include "NPU/Encoding/Program.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

namespace {

cl::opt<std::string> inputFilename(cl::Positional,
                                   cl::desc("<input .mlir file>"),
                                   cl::init("-"));

cl::opt<std::string> outputFilename("o", cl::desc("Write the .nbin here"),
                                    cl::value_desc("filename"),
                                    cl::init("-"));

cl::opt<bool> stripDebug(
    "strip-debug",
    cl::desc("Write an empty debug section. A stripped binary is legal and "
             "this is what produces one."),
    cl::init(false));

} // namespace

int main(int argc, char **argv) {
  InitLLVM lifetime(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "npu-translate: allocated npuisa IR to a .nbin binary\n");

  DialectRegistry registry;
  registry.insert<npu::NPUDialect, npuisa::NPUISADialect, func::FuncDialect,
                  memref::MemRefDialect, arith::ArithDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer =
      MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!buffer) {
    WithColor::error(errs(), "npu-translate")
        << "cannot read " << inputFilename << ": "
        << buffer.getError().message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*buffer), SMLoc());
  SourceMgrDiagnosticHandler diagnostics(sourceMgr, &context);

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  nbin::Program program;
  if (failed(nbin::encodeModule(*module, program, stripDebug)))
    return 1;

  // The encoder's own output goes through the same validator every file does.
  if (std::optional<nbin::ProgramError> failure = program.validate()) {
    WithColor::error(errs(), "npu-translate")
        << "the encoder produced a program that does not validate: "
        << failure->toString() << "\n";
    errs() << "npu-translate: this is a defect in the encoder rather than in "
              "the input, and no file has been written.\n";
    return 1;
  }

  std::vector<uint8_t> bytes = program.encode();

  // Only now is anything created on disk.
  if (outputFilename == "-") {
    outs().write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return 0;
  }

  std::error_code fileError;
  raw_fd_ostream output(outputFilename, fileError, sys::fs::OF_None);
  if (fileError) {
    WithColor::error(errs(), "npu-translate")
        << "cannot open " << outputFilename << ": " << fileError.message()
        << "\n";
    return 1;
  }
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error()) {
    WithColor::error(errs(), "npu-translate")
        << "cannot write " << outputFilename << "\n";
    return 1;
  }
  return 0;
}
