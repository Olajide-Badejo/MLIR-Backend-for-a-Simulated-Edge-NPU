// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// npu-sim, over a file the whole pipeline produced.
//
// The numerics of every kernel are asserted in NPUSimulatorTests against hand
// computed values. What this file asserts is the **tool's** contract with the
// format, which is Section 9.3's three obligations and which no unit test can
// reach because none of them goes through a command line:
//
//   STATS     a program the compiler produced runs, and the statistics come out
//             with stats.instructions among them, which Section 10.2 makes the
//             only instruction count anywhere in this project.
//   JSON      the same run's statistics as data, which is what a caller reads.
//             Section 16.2's rule that nothing is scraped out of human readable
//             text applies here for a specific reason: the one number nothing
//             may guess at is stats.instructions, and it is exactly the number a
//             text parser would guess at.
//   SERIAL    the single port flag of Section 5.5, which reports no overlap.
//   COUNT     one --input per declared input region, refused with **both**
//             numbers when the counts disagree.
//   SIZE      the input file's size against the declared region, likewise.
//   OUTCOUNT  one --output per declared region, for the same reason.
//   KERNEL    --kernel-info answers D-0047 from outside the process, needs no
//             program, and exits 0. The value is not asserted, because whether
//             a build has OpenMP is a property of the machine it was configured
//             on and CI runs both shapes; what is asserted is that the tool
//             answers in the two words a caller parses, so a rename is caught
//             here rather than by a harness quietly reading "no" forever.
//
// And the claim that needs no CHECK line to state but does need one to prove:
// **all** outputs are written, not just the first. The two output files below
// are both asked for and both measured.

// RUN: npu-opt %s --npu-lower-to-npuisa --npu-allocate-scratchpad \
// RUN:   | npu-translate -o %t.nbin
// RUN: head -c 128 /dev/zero > %t.in
// RUN: npu-sim %t.nbin --input %t.in --output %t.out0 --output %t.out1 \
// RUN:   --json-stats %t.stats.json | FileCheck %s --check-prefix=STATS
// RUN: FileCheck %s --check-prefix=JSON < %t.stats.json

// RUN: wc -c < %t.out0 | FileCheck %s --check-prefix=SIZE0
// RUN: wc -c < %t.out1 | FileCheck %s --check-prefix=SIZE1

// RUN: npu-sim %t.nbin --input %t.in --output %t.out0 --output %t.out1 \
// RUN:   --single-port | FileCheck %s --check-prefix=SERIAL

// RUN: not npu-sim %t.nbin --output %t.out0 --output %t.out1 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COUNT

// RUN: head -c 64 /dev/zero > %t.short.in
// RUN: not npu-sim %t.nbin --input %t.short.in --output %t.out0 \
// RUN:   --output %t.out1 2>&1 | FileCheck %s --check-prefix=SIZE

// RUN: not npu-sim %t.nbin --input %t.in --output %t.out0 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OUTCOUNT

// No program argument at all, which is the point: it reports a property of the
// build rather than of a run.
// RUN: npu-sim --kernel-info | FileCheck %s --check-prefix=KERNEL

// STATS:      instructions: 6
// STATS-NEXT: cycles:
// STATS-NEXT: dma cycles:
// STATS-NEXT: compute cycles:
// STATS-NEXT: overlap fraction:
// STATS-NEXT: dram bytes read: 128
// STATS-NEXT: dram bytes written: 256
// STATS:      macs: 0
// STATS-NEXT: int8 macs: 0

// The same three numbers, from the same run, as data. The keys are the text
// labels with their spaces turned into underscores, so a field that gained a
// spelling in one printer and not the other shows up here.
// JSON-DAG: "instructions": 6
// JSON-DAG: "dram_bytes_read": 128
// JSON-DAG: "dram_bytes_written": 256
// JSON-DAG: "reached_halt": true
// JSON-DAG: "single_port": false

// The two outputs are 2x4x4 f32 each, which is 128 bytes each. Both files are
// measured, because "writes all outputs" is a claim about the second one.
// SIZE0: 128
// SIZE1: 128

// SERIAL: overlap fraction: 0.0000

// COUNT: error
// COUNT-SAME: declares 1 input regions and 0 --input arguments were given

// SIZE: error
// SIZE-SAME: input region 0 is 128 bytes and the file supplied is 64

// OUTCOUNT: error
// OUTCOUNT-SAME: declares 2 output regions and 1 --output arguments were given

// KERNEL:      kernel openmp: {{yes|no}}
// KERNEL-NEXT: kernel threads: {{[0-9]+}}

func.func @two_outputs(%x: tensor<2x4x4xf32>)
    -> (tensor<2x4x4xf32>, tensor<2x4x4xf32>) {
  %d0 = tensor.empty() : tensor<2x4x4xf32>
  %r = npu.relu ins(%x : tensor<2x4x4xf32>) outs(%d0 : tensor<2x4x4xf32>)
       -> tensor<2x4x4xf32>
  %d1 = tensor.empty() : tensor<2x4x4xf32>
  %a = npu.add ins(%r, %x : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
               outs(%d1 : tensor<2x4x4xf32>)
       -> tensor<2x4x4xf32>
  return %r, %a : tensor<2x4x4xf32>, tensor<2x4x4xf32>
}
