//===- npu-sim.cpp - running a .nbin --------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Loads a `.nbin`, runs it, writes its outputs, and prints what it measured.
//
// Section 9.3 puts three obligations on this tool specifically and each one is
// a refusal rather than a best effort:
//
//   1. The size of every input file is compared against the declared input
//      region and a mismatch is refused. `Simulator::loadInput` does the
//      comparison, so every caller gets it and not only this one.
//   2. One `--input` per declared input region. A count mismatch is refused
//      with a message naming **both** numbers, because "wrong number of
//      inputs" sends the reader to count them by hand.
//   3. **All** outputs are written, not just the first.
//
// The exit code says which of two things happened: 0 for a run that reached the
// end of its program, 1 for a refusal at any point. A trap inside the run is a
// refusal, because the outputs after one are whatever the skipped writes left
// behind and handing those to a caller as an answer would be worse than
// handing them nothing.
//
// **`--stats-json` is how a program reads the statistics, and it exists so that
// nothing scrapes them out of the text below.** Section 16.2 states the rule for
// the instrumentation and it applies here for the same reason: a caller that
// parsed `printStats`'s output would be one relabelled line away from reading a
// number as zero, and Section 10.2 makes `stats.instructions` the only
// instruction count in this project, so the one number nothing may guess at is
// exactly the one a text parser would guess at. The JSON is written after the
// run succeeds and never on a refusal, because a statistics file whose presence
// did not mean the run finished would be a file every caller had to check twice.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/Simulator.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<program .nbin>"),
                                   cl::init("-"));

cl::list<std::string> inputFiles(
    "input", cl::desc("Raw bytes for one declared input region, in order"),
    cl::value_desc("filename"));

cl::list<std::string> outputFiles(
    "output", cl::desc("Where to write one declared output region, in order"),
    cl::value_desc("filename"));

cl::opt<bool> singlePort(
    "single-port",
    cl::desc("Put every instruction on one port, per Section 5.5's "
             "reproducibility flag"),
    cl::init(false));

cl::opt<bool> quiet("quiet", cl::desc("Do not print the statistics"),
                    cl::init(false));

// `--json-stats` and not `--stats-json`, which is the name this flag wanted and
// cannot have: LLVM's Support library registers `--stats-json` itself, for the
// `llvm::Statistic` counters, and every tool that links Support inherits it. The
// collision is an abort inside `ParseCommandLineOptions` at the first run, which
// is a loud failure and not a silent one, but the reason for the odd name
// belongs beside the name rather than in a commit message.
cl::opt<std::string> statsJsonFilename(
    "json-stats",
    cl::desc("Write the simulator's statistics here as JSON, for a caller that "
             "reads them rather than a reader who looks at them"),
    cl::value_desc("filename"));

/// Prints `Stats` as one field per line.
///
/// Section 10.2 makes `stats.instructions` the only instruction count anywhere
/// in this project, so it is printed first and it is printed by name. The four
/// occupancy fields are printed beside `macs` rather than instead of it, so a
/// reader can reconstruct the cycle charge without reverse engineering it and
/// the energy path never sees a scaled number.
void printStats(raw_ostream &out, const nbin::Stats &stats) {
  out << "instructions: " << stats.instructions << "\n";
  out << "cycles: " << format("%.4f", stats.cycles) << "\n";
  out << "dma cycles: " << format("%.4f", stats.dmaCycles) << "\n";
  out << "compute cycles: " << format("%.4f", stats.computeCycles) << "\n";
  out << "overlap fraction: " << format("%.4f", stats.overlapFraction) << "\n";
  out << "dram bytes read: " << stats.dramBytesRead << "\n";
  out << "dram bytes written: " << stats.dramBytesWritten << "\n";
  out << "scratchpad elements read: " << stats.scratchpadElementsRead << "\n";
  out << "scratchpad elements written: " << stats.scratchpadElementsWritten
      << "\n";
  out << "macs: " << stats.macs << "\n";
  out << "int8 macs: " << stats.int8Macs << "\n";
  out << "effective macs: " << format("%.4f", stats.effectiveMacs) << "\n";
  out << "utilization: " << format("%.6f", stats.utilization) << "\n";
  out << "delta: " << format("%.6f", stats.delta) << "\n";
}

/// The same fields as `printStats`, as JSON, under the same names.
///
/// The key for each field is the text label with its spaces turned into
/// underscores, so there is one vocabulary here rather than two. A field
/// present in one printer and absent from the other is the drift this pairing
/// exists to make visible, and `test/Simulator/npu-sim.mlir` asserts both
/// printers over the same run.
///
/// `reached_halt` is here and not in the text form, because the text form says
/// it on stderr as a note and a note is not a field. A caller deciding whether
/// to trust a cycle count needs to know whether the machine stopped because the
/// program said so or because it ran out of program.
json::Object statsAsJson(const nbin::Stats &stats, bool reachedHalt,
                         bool singlePortRun) {
  return json::Object{
      {"instructions", stats.instructions},
      {"cycles", stats.cycles},
      {"dma_cycles", stats.dmaCycles},
      {"compute_cycles", stats.computeCycles},
      {"overlap_fraction", stats.overlapFraction},
      {"dram_bytes_read", stats.dramBytesRead},
      {"dram_bytes_written", stats.dramBytesWritten},
      {"scratchpad_elements_read", stats.scratchpadElementsRead},
      {"scratchpad_elements_written", stats.scratchpadElementsWritten},
      {"macs", stats.macs},
      {"int8_macs", stats.int8Macs},
      {"effective_macs", stats.effectiveMacs},
      {"utilization", stats.utilization},
      {"delta", stats.delta},
      {"reached_halt", reachedHalt},
      {"single_port", singlePortRun},
  };
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM lifetime(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "npu-sim: run a .nbin binary\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer =
      inputFilename == "-" ? MemoryBuffer::getSTDIN()
                           : MemoryBuffer::getFile(inputFilename, false, false);
  if (!buffer) {
    WithColor::error(errs(), "npu-sim")
        << "cannot read " << inputFilename << ": "
        << buffer.getError().message() << "\n";
    return 1;
  }

  ArrayRef<uint8_t> bytes(
      reinterpret_cast<const uint8_t *>((*buffer)->getBufferStart()),
      (*buffer)->getBufferSize());

  nbin::Program program;
  if (std::optional<nbin::ProgramError> failure =
          nbin::Program::decode(bytes, program)) {
    WithColor::error(errs(), "npu-sim")
        << "cannot load " << inputFilename << ": " << failure->toString()
        << "\n";
    return 1;
  }

  // One --input per declared region, and one --output per declared region. Both
  // numbers in both messages: a reader who is told only that the count is wrong
  // has to go and count.
  if (inputFiles.size() != program.inputs.size()) {
    WithColor::error(errs(), "npu-sim")
        << "this program declares " << program.inputs.size()
        << " input regions and " << inputFiles.size()
        << " --input arguments were given. One --input per declared region, in "
           "order.\n";
    return 1;
  }
  if (outputFiles.size() != program.outputs.size()) {
    WithColor::error(errs(), "npu-sim")
        << "this program declares " << program.outputs.size()
        << " output regions and " << outputFiles.size()
        << " --output arguments were given. Every output is written, so one "
           "--output per declared region, in order.\n";
    return 1;
  }

  nbin::Simulator simulator(program);

  for (size_t index = 0; index < inputFiles.size(); ++index) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> data =
        MemoryBuffer::getFile(inputFiles[index], false, false);
    if (!data) {
      WithColor::error(errs(), "npu-sim")
          << "cannot read " << inputFiles[index] << ": "
          << data.getError().message() << "\n";
      return 1;
    }
    ArrayRef<uint8_t> raw(
        reinterpret_cast<const uint8_t *>((*data)->getBufferStart()),
        (*data)->getBufferSize());
    std::string failure;
    if (!simulator.loadInput(index, raw, failure)) {
      WithColor::error(errs(), "npu-sim")
          << inputFiles[index] << ": " << failure << "\n";
      return 1;
    }
  }

  nbin::SimOptions options;
  options.singlePort = singlePort;
  const nbin::SimResult result = simulator.run(options);

  if (!result.ok()) {
    WithColor::error(errs(), "npu-sim") << *result.error << "\n";
    return 1;
  }

  // The Phase P7 decision on a program that runs out of instructions: the
  // machine stops, exactly as a HALT would have stopped it, and says so. It is
  // not an error, because docs/ISA_MANUAL.md already says a machine with no
  // branches stops when it runs out of straight line code, and because
  // refusing here would enforce at run time a rule the validator was
  // deliberately not given.
  if (!result.reachedHalt)
    WithColor::note(errs(), "npu-sim")
        << "the program ran out of instructions without a HALT. The machine "
           "stopped, which is what the instruction set says it does, and the "
           "statistics below are for the instructions that ran.\n";

  for (size_t index = 0; index < outputFiles.size(); ++index) {
    ArrayRef<uint8_t> data = simulator.outputBytes(index);
    std::error_code fileError;
    raw_fd_ostream out(outputFiles[index], fileError, sys::fs::OF_None);
    if (fileError) {
      WithColor::error(errs(), "npu-sim")
          << "cannot open " << outputFiles[index] << ": "
          << fileError.message() << "\n";
      return 1;
    }
    out.write(reinterpret_cast<const char *>(data.data()), data.size());
  }

  if (!statsJsonFilename.empty()) {
    std::error_code fileError;
    raw_fd_ostream statsOut(statsJsonFilename, fileError, sys::fs::OF_Text);
    if (fileError) {
      WithColor::error(errs(), "npu-sim")
          << "cannot open " << statsJsonFilename << ": " << fileError.message()
          << "\n";
      return 1;
    }
    statsOut << llvm::formatv("{0:2}",
                              json::Value(statsAsJson(result.stats,
                                                      result.reachedHalt,
                                                      options.singlePort)))
             << "\n";
    statsOut.close();
    if (statsOut.has_error()) {
      WithColor::error(errs(), "npu-sim")
          << "cannot write " << statsJsonFilename << "\n";
      return 1;
    }
  }

  if (!quiet)
    printStats(outs(), result.stats);
  return 0;
}
