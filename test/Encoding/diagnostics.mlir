// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Everything `npu-translate` refuses, by name, and the file it does not write.
//
// The cases live in `Inputs/` rather than in this file because each of them is
// a whole module and `npu-translate` reads one module per run. There is no
// `--split-input-file` here and there should not be: the tool's contract is one
// file, one program.
//
// **Every case asserts two things.** That the diagnostic names the refusal, and
// that no output file exists afterwards. The second is the half that would rot
// silently: the roadmap entry for this phase asks for the output file not to be
// created before the encode result is known, and a tool that opened the file
// first and failed later would satisfy every message check in this file while
// leaving a zero byte `.nbin` behind on each one. A build system that treats an
// existing file as an up to date one then carries that emptiness forward.

// RUN: not npu-translate %S/Inputs/two-functions.mlir -o %t.two.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TWO
// RUN: not test -e %t.two.nbin
// TWO: error: the module holds 2 functions and a .nbin holds one program
// TWO: note: function @first
// TWO: note: function @second

// RUN: not npu-translate %S/Inputs/no-arg-kind.mlir -o %t.argkind.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARGKIND
// RUN: not test -e %t.argkind.nbin
// ARGKIND: error: argument 0 carries no `npuisa.arg` attribute

// RUN: not npu-translate %S/Inputs/out-before-in.mlir -o %t.order.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ORDER
// RUN: not test -e %t.order.nbin
// ORDER: error: argument 1 is an input and follows an output

// RUN: not npu-translate %S/Inputs/no-scratchpad-bytes.mlir -o %t.bytes.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BYTES
// RUN: not test -e %t.bytes.nbin
// BYTES: error: the function carries no `npuisa.scratchpad_bytes` attribute

// RUN: not npu-translate %S/Inputs/unallocated.mlir -o %t.unalloc.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNALLOC
// RUN: not test -e %t.unalloc.nbin
// UNALLOC: error: this scratchpad buffer is not a view over the allocator's arena

// RUN: not npu-translate %S/Inputs/returns-a-value.mlir -o %t.returns.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RETURNS
// RUN: not test -e %t.returns.nbin
// RETURNS: error: cannot encode a function that returns a value

// RUN: not npu-translate %S/Inputs/unencodable-op.mlir -o %t.unknown.nbin 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNKNOWN
// RUN: not test -e %t.unknown.nbin
// UNKNOWN: error: cannot encode this operation
