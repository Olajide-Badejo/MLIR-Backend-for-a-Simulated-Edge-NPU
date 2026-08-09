// An npu op that was never lowered to npuisa reaches the encoder, which has no
// case for it. npu-translate has to refuse: exit nonzero, say why, and leave no
// output file. It used to print the error, write the .nbin, and exit 0, so the
// build succeeded and the program it produced was missing the relu.

// RUN: rm -f %t.nbin
// RUN: not npu-translate %s -o %t.nbin 2>&1 | FileCheck %s
// RUN: test ! -f %t.nbin

// CHECK: cannot encode unexpected op

func.func @main(%x: tensor<1x2xf32>) -> tensor<1x2xf32>
    attributes {npuisa.scratchpad_bytes = 16 : i64} {
  %0 = npu.relu %x : tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
