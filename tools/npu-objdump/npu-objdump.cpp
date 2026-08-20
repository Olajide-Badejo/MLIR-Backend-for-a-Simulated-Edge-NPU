//===- npu-objdump.cpp - disassembling a .nbin ---------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Prints a `.nbin` as text.
//
// **It decodes without validating, on purpose.** Section 9.2 gives
// `decodeUnvalidated()` exactly one reason to exist: so that this tool can
// dump a suspect file. A disassembler that refused every file the validator
// refused would be useless at the moment it was most needed, which is the
// moment somebody is holding a file that does not validate and wants to know
// why.
//
// What it does not do is read past the end of a buffer to show somebody what
// is there. A file it cannot frame is refused, because a truncated section has
// no well defined contents and inventing some would be the bug this whole
// subsystem exists to prevent.
//
// The exit code says which of the three things happened: 0 for a file that
// decoded, whether or not it validated; 1 for a file that could not be framed
// or read at all. A file that decoded but did not validate is reported through
// the warning block at the top of the output rather than through the exit
// code, because the output is the point.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"
#include "NPU/Encoding/Program.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input .nbin>"),
                                   cl::init("-"));

cl::opt<std::string> outputFilename("o", cl::desc("Write the listing here"),
                                    cl::value_desc("filename"),
                                    cl::init("-"));

} // namespace

int main(int argc, char **argv) {
  InitLLVM lifetime(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "npu-objdump: disassemble a .nbin binary\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer =
      inputFilename == "-" ? MemoryBuffer::getSTDIN()
                           : MemoryBuffer::getFile(inputFilename, false, false);
  if (!buffer) {
    WithColor::error(errs(), "npu-objdump")
        << "cannot read " << inputFilename << ": "
        << buffer.getError().message() << "\n";
    return 1;
  }

  ArrayRef<uint8_t> bytes(
      reinterpret_cast<const uint8_t *>((*buffer)->getBufferStart()),
      (*buffer)->getBufferSize());

  nbin::Program program;
  if (std::optional<nbin::ProgramError> failure =
          nbin::Program::decodeUnvalidated(bytes, program)) {
    WithColor::error(errs(), "npu-objdump")
        << "cannot decode " << inputFilename << ": " << failure->toString()
        << "\n";
    return 1;
  }

  std::optional<nbin::ProgramError> failure = program.validate();
  std::string listing = nbin::disassemble(program, failure);

  if (outputFilename == "-") {
    outs() << listing;
    return 0;
  }

  std::error_code fileError;
  raw_fd_ostream output(outputFilename, fileError, sys::fs::OF_None);
  if (fileError) {
    WithColor::error(errs(), "npu-objdump")
        << "cannot open " << outputFilename << ": " << fileError.message()
        << "\n";
    return 1;
  }
  output << listing;
  return 0;
}
