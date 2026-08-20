// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Offset assignment: where each buffer lands, and why.
//
// Every offset in this file is checked by value, with the arithmetic written
// out beside it. A test that only checked that *an* offset appeared would pass
// against an allocator that gave every buffer offset zero, which is the one
// wrong answer that produces a program that runs and computes nonsense.
//
// Both offset assignment strategies of Section 13.1 run over the same file,
// under three run lines and three check prefixes. The fragmentation case is the
// reason for that shape: it is one program on which the two strategies
// disagree, and a file that ran only one of them could not show the
// disagreement at all.
//
// The budget is pinned on every run line rather than left at the default of one
// mebibyte. A 1 MiB default would place every buffer here without ever
// exercising a boundary, and a test whose budget cannot bind is a test of the
// happy path only.

// RUN: npu-opt %s --npu-allocate-scratchpad="strategy=pack budget=8192" \
// RUN:   | FileCheck %s --check-prefixes=CHECK,PACK
// RUN: npu-opt %s --npu-allocate-scratchpad="strategy=interval budget=8192" \
// RUN:   | FileCheck %s --check-prefixes=CHECK,INTERVAL
// RUN: npu-opt %s --npu-allocate-scratchpad="strategy=pack budget=8192" \
// RUN:   | FileCheck %s --check-prefix=NOALLOC
// RUN: npu-opt %s --npu-allocate-scratchpad="strategy=pack budget=8192" \
// RUN:   --mlir-pass-statistics 2>&1 | FileCheck %s --check-prefix=STATS

// The claim that needs the whole module rather than one function: **no
// scratchpad buffer allocation survives anywhere.** Every one of them is a view
// into the arena, which is Section 8's rule that the offset is an SSA operand.
// It is its own run line because a `CHECK-NOT` inside a labelled block only
// covers the gap between two positive matches, and this assertion is about the
// gaps as well as about them.
//
// NOALLOC: module
// NOALLOC-NOT: memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
// NOALLOC-NOT: memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
// NOALLOC-NOT: memref.alloc() : memref<17xf32, #npu.scratchpad>
// NOALLOC-NOT: memref.alloc() : memref<8xf32, #npu.scratchpad>

// =============================================================================
// It fits, and a dead buffer's bytes come back.
//
// Four operations, three buffers of 1 x 8 x 4 x 4 f32, which is 128 elements at
// 4 bytes each, so 512 bytes apiece.
//
//   %a  live [0, 3]   loaded, then read by the first relu
//   %b  live [2, 5]   written by the first relu, read by the second
//   %c  live [4, 7]   written by the second relu, then stored
//
// %a and %c never share an index, so %c takes %a's offset back. That is the
// whole point of computing liveness rather than laying buffers out end to end:
// end to end would need 1536 bytes for a program that fits in 1024.
// =============================================================================

// CHECK-LABEL: func.func @fits_in_the_budget
// CHECK-SAME:    npuisa.fragmentation_ratio = 1.000000e+00 : f64
// CHECK-SAME:    npuisa.scratchpad_budget = 8192 : i64
// CHECK-SAME:    npuisa.scratchpad_bytes = 1024 : i64
// CHECK-SAME:    npuisa.scratchpad_peak_bytes = 1024 : i64
// CHECK-SAME:    npuisa.spill_count = 0 : i64
// CHECK-SAME:    npuisa.spill_dma_count = 0 : i64
func.func @fits_in_the_budget(%in: memref<1x8x4x4xf32, #npu.dram>,
                              %out: memref<1x8x4x4xf32, #npu.dram>) {
  // The arena is exactly the high water mark, not the budget. 1024 bytes is
  // two buffers of 512, which is the most that are ever live at once.
  // CHECK: %[[ARENA:.*]] = memref.alloc()
  // CHECK-SAME: {alignment = 64 : i64, npuisa.scratchpad_arena}
  // CHECK-SAME: : memref<1024xi8, #npu.scratchpad>

  // %a at 0.
  // CHECK: %[[O0:.*]] = arith.constant 0 : index
  // CHECK: %[[A:.*]] = memref.view %[[ARENA]][%[[O0]]][]
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK: npuisa.dma_load %arg0, %[[A]]
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>

  // %b at 512, because %a is still live.
  // CHECK: %[[O512:.*]] = arith.constant 512 : index
  // CHECK: %[[B:.*]] = memref.view %[[ARENA]][%[[O512]]][]
  %b = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK: npuisa.relu ins(%[[A]] {{.*}}) outs(%[[B]]
  npuisa.relu ins(%a : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x8x4x4xf32, #npu.scratchpad>)

  // %c back at 0: %a died at the relu above and its bytes are free again.
  // CHECK: %[[O0B:.*]] = arith.constant 0 : index
  // CHECK: %[[C:.*]] = memref.view %[[ARENA]][%[[O0B]]][]
  %c = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK: npuisa.relu ins(%[[B]] {{.*}}) outs(%[[C]]
  npuisa.relu ins(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%c : memref<1x8x4x4xf32, #npu.scratchpad>)
  // CHECK: npuisa.dma_store %[[C]], %arg1
  npuisa.dma_store %c, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}

// =============================================================================
// No reuse while live, which is the case Section 13.1 says would be silently
// wrong rather than loudly wrong.
//
// Three buffers of 256 bytes, 1 x 4 x 4 x 4 f32 at 64 elements.
//
//   %a  live [0, 3]
//   %b  live [1, 5]   overlaps %a at indices 1 to 3
//   %c  live [4, 6]   overlaps %b, disjoint from %a
//
// So %a and %b must differ and %c may take %a's offset. Both halves are checked,
// because an allocator that never reused would pass the first half alone.
// =============================================================================

// CHECK-LABEL: func.func @no_reuse_while_live
// CHECK-SAME:    npuisa.scratchpad_bytes = 512 : i64
// CHECK-SAME:    npuisa.scratchpad_peak_bytes = 512 : i64
func.func @no_reuse_while_live(%in: memref<1x4x4x4xf32, #npu.dram>,
                               %out: memref<1x4x4x4xf32, #npu.dram>) {
  // CHECK: %[[ARENA:.*]] = memref.alloc()
  // CHECK-SAME: : memref<512xi8, #npu.scratchpad>
  %a = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x4x4x4xf32, #npu.dram> to memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<1x4x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x4x4x4xf32, #npu.scratchpad>)
  %c = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%b : memref<1x4x4x4xf32, #npu.scratchpad>)
              outs(%c : memref<1x4x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %c, %out
    : memref<1x4x4x4xf32, #npu.scratchpad> to memref<1x4x4x4xf32, #npu.dram>
  return
}

// =============================================================================
// Alignment. Every offset is a multiple of 64, which is the width of a row of
// the 16 by 16 array at f32.
//
// The first buffer is 17 f32, which is 68 bytes, and the second cannot start at
// 68. It starts at 128, because 68 rounds up to the next multiple of 64. That
// is the only case in this file whose expected offset is not also the sum of
// the sizes before it, which is why it is here.
// =============================================================================

// CHECK-LABEL: func.func @alignment_rounds_the_next_offset_up
// CHECK-SAME:    npuisa.scratchpad_bytes = 196 : i64
// CHECK-SAME:    npuisa.scratchpad_peak_bytes = 136 : i64
func.func @alignment_rounds_the_next_offset_up(%in: memref<17xf32, #npu.dram>,
                                               %out: memref<17xf32, #npu.dram>) {
  // 128 for the offset plus 68 for the buffer is 196 bytes of arena.
  // CHECK: %[[ARENA:.*]] = memref.alloc()
  // CHECK-SAME: : memref<196xi8, #npu.scratchpad>
  %a = memref.alloc() : memref<17xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<17xf32, #npu.scratchpad>
  // CHECK: %[[O0:.*]] = arith.constant 0 : index
  // CHECK: %[[A:.*]] = memref.view %[[ARENA]][%[[O0]]][]
  // CHECK: %[[O128:.*]] = arith.constant 128 : index
  // CHECK: %[[B:.*]] = memref.view %[[ARENA]][%[[O128]]][]
  npuisa.dma_load %in, %a
    : memref<17xf32, #npu.dram> to memref<17xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<17xf32, #npu.scratchpad>)
              outs(%b : memref<17xf32, #npu.scratchpad>)
  npuisa.dma_store %b, %out
    : memref<17xf32, #npu.scratchpad> to memref<17xf32, #npu.dram>
  return
}

// =============================================================================
// Fragmentation, and the two strategies disagreeing about it.
//
//   %a  256 bytes, live [0, 3]
//   %b  256 bytes, live [1, 5]
//   %c  512 bytes, live [4, 6]
//
// The sweep line peak is 768, at index 4, where %b and %c are both live. Both
// strategies see the same peak, because the peak is a property of the program
// and not of the placement.
//
// `pack` places largest first: %c at 0, then %b at 512, then %a back at 0
// because %a and %c never share an index. High water 768, so the fragmentation
// ratio is exactly 1 and the placement is optimal.
//
// `interval` places in definition order: %a at 0, %b at 256 because it overlaps
// %a, and %c cannot start at 0 (it would run into %b at 256) so it starts at
// 512. High water 1024 against a peak of 768, a fragmentation ratio of 4/3.
// The 256 bytes below %b are a hole nothing fits in.
//
// **That gap is the whole reason the spill trigger is offset assignment
// failing rather than the peak exceeding the budget.** At a budget of 1000 this
// program's peak fits and its interval placement does not; `spill-heuristic.mlir`
// carries that case.
// =============================================================================

// CHECK-LABEL:      func.func @fragmentation
// PACK-SAME:          npuisa.fragmentation_ratio = 1.000000e+00 : f64
// INTERVAL-SAME:      npuisa.fragmentation_ratio = 1.3333333333333333 : f64
// PACK-SAME:          npuisa.scratchpad_bytes = 768 : i64
// INTERVAL-SAME:      npuisa.scratchpad_bytes = 1024 : i64
// CHECK-SAME:         npuisa.scratchpad_peak_bytes = 768 : i64
func.func @fragmentation(%in: memref<1x4x4x4xf32, #npu.dram>,
                         %out: memref<1x8x4x4xf32, #npu.dram>) {
  // PACK:     memref.alloc() {{.*}} : memref<768xi8, #npu.scratchpad>
  // INTERVAL: memref.alloc() {{.*}} : memref<1024xi8, #npu.scratchpad>
  %a = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  // %a at 0 under both, %b at 512 under pack and 256 under interval.
  // CHECK:        arith.constant 0 : index
  // PACK:         arith.constant 512 : index
  // INTERVAL:     arith.constant 256 : index
  npuisa.dma_load %in, %a
    : memref<1x4x4x4xf32, #npu.dram> to memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<1x4x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x4x4x4xf32, #npu.scratchpad>)
  // %c back at 0 under pack, at 512 under interval.
  // PACK:         arith.constant 0 : index
  // INTERVAL:     arith.constant 512 : index
  %c = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.concat ins(%b, %b : memref<1x4x4x4xf32, #npu.scratchpad>,
                             memref<1x4x4x4xf32, #npu.scratchpad>)
                outs(%c : memref<1x8x4x4xf32, #npu.scratchpad>) {axis = 1 : i64}
  npuisa.dma_store %c, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}

// =============================================================================
// A buffer whose type carries a strided layout map, which is what an NHWC
// tensor lowers to.
//
// `memref.view` requires an identity layout on its result, so the offset is
// materialised as a view at the buffer's extents and a `memref.reinterpret_cast`
// puts the layout back on top of it. The bytes are the same bytes: a permutation
// layout spans exactly the contiguous extent its shape does, which is why the
// arena is 1536 bytes for two buffers of 1 x 3 x 8 x 8 f32 at 768 each.
// =============================================================================

// CHECK-LABEL: func.func @a_strided_buffer_keeps_its_layout
// CHECK-SAME:    npuisa.scratchpad_bytes = 1536 : i64
func.func @a_strided_buffer_keeps_its_layout(
    %out: memref<1x3x8x8xf32, #npu.dram>) {
  // CHECK: %[[ARENA:.*]] = memref.alloc()
  // CHECK: %[[O0:.*]] = arith.constant 0 : index
  // CHECK: %[[V:.*]] = memref.view %[[ARENA]][%[[O0]]][]
  // CHECK-SAME: to memref<1x3x8x8xf32, #npu.scratchpad>
  // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[V]] to
  // CHECK-SAME: offset: [0], sizes: [1, 3, 8, 8], strides: [192, 1, 24, 3]
  // CHECK-SAME: to memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>
  %a = memref.alloc()
     : memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>
  %b = memref.alloc() : memref<1x3x8x8xf32, #npu.scratchpad>
  // CHECK: npuisa.relu ins(%[[CAST]]
  npuisa.relu
      ins(%a : memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>)
      outs(%b : memref<1x3x8x8xf32, #npu.scratchpad>)
  npuisa.dma_store %b, %out
    : memref<1x3x8x8xf32, #npu.scratchpad> to memref<1x3x8x8xf32, #npu.dram>
  return
}

// =============================================================================
// The rank 1 channel broadcast of ADR 0005, which the lowering emits as a
// `memref.reinterpret_cast` over the loaded rank 1 buffer.
//
// Two things are asserted. The allocator's view goes **underneath** the
// existing cast, so the cast is left exactly as the lowering wrote it and the
// broadcast semantics are untouched. And the rank 1 buffer stays live for as
// long as the cast is read: liveness follows views, so the 32 bytes of scale
// are not handed to somebody else while the multiply is still reading them.
// =============================================================================

// The peak is a property of the program, so both strategies see 1056: two 512
// byte buffers and the 32 byte scale, all live at the multiply. The high water
// mark is a property of the placement and they differ, which is the same
// disagreement the fragmentation case above isolates.
//
// The `-SAME` directives are in the order the attribute dictionary prints,
// because they match left to right along one line and a later directive cannot
// match text an earlier one has already scanned past.
//
// CHECK-LABEL:   func.func @a_broadcast_view_sits_above_the_allocator_view
// PACK-SAME:       npuisa.scratchpad_bytes = 1056 : i64
// INTERVAL-SAME:   npuisa.scratchpad_bytes = 1088 : i64
// CHECK-SAME:      npuisa.scratchpad_peak_bytes = 1056 : i64
func.func @a_broadcast_view_sits_above_the_allocator_view(
    %in: memref<1x8x4x4xf32, #npu.dram>, %scale: memref<8xf32, #npu.dram>,
    %out: memref<1x8x4x4xf32, #npu.dram>) {
  // CHECK: %[[ARENA:.*]] = memref.alloc()
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  // The scale buffer is 8 f32, so 32 bytes. Under `pack` it lands above the two
  // 512 byte buffers at 1024, and 1024 plus 32 is the 1056 byte arena.
  // PACK: %[[O1024:.*]] = arith.constant 1024 : index
  // CHECK: %[[S:.*]] = memref.view %[[ARENA]][%{{.*}}][] : {{.*}} to memref<8xf32, #npu.scratchpad>
  %s = memref.alloc() : memref<8xf32, #npu.scratchpad>
  npuisa.dma_load %scale, %s
    : memref<8xf32, #npu.dram> to memref<8xf32, #npu.scratchpad>
  // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[S]] to
  // CHECK-SAME: strides: [0, 1, 0, 0]
  %view = memref.reinterpret_cast %s to offset: [0], sizes: [1, 8, 4, 4],
                                        strides: [0, 1, 0, 0]
      : memref<8xf32, #npu.scratchpad>
     to memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
  %d = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK: npuisa.mul ins(%{{.*}}, %[[CAST]]
  npuisa.mul ins(%a, %view
      : memref<1x8x4x4xf32, #npu.scratchpad>,
        memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>)
      outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %d, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}

// =============================================================================
// Where it does not fire. Section 12's negative test rule: a pass with only
// positive tests is not adequately tested, because a pass that fired
// unconditionally would pass them all.
// =============================================================================

// A function that has already been allocated is left exactly as it is. This is
// the idempotence guard and it is not decoration: the arena is itself a
// scratchpad allocation, so a second run without the guard would allocate an
// arena for the arena and grow the program every time the pipeline ran.
//
// CHECK-LABEL: func.func @an_already_allocated_function_is_untouched
// CHECK-SAME:    npuisa.scratchpad_bytes = 4096 : i64
// CHECK-NEXT:  %[[ARENA:.*]] = memref.alloc()
// CHECK-SAME:    : memref<4096xi8, #npu.scratchpad>
// CHECK-NEXT:  %[[OFF:.*]] = arith.constant 128 : index
// CHECK-NEXT:  %[[V:.*]] = memref.view %[[ARENA]][%[[OFF]]][]
// CHECK-NEXT:  npuisa.dma_store %[[V]], %arg0
// CHECK-NEXT:  return
// CHECK-NOT:   memref.view
func.func @an_already_allocated_function_is_untouched(
    %out: memref<1x8x4x4xf32, #npu.dram>)
    attributes {npuisa.scratchpad_bytes = 4096 : i64} {
  %arena = memref.alloc() : memref<4096xi8, #npu.scratchpad>
  %off = arith.constant 128 : index
  %v = memref.view %arena[%off][]
     : memref<4096xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_store %v, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}

// A function with nothing to allocate gets the attributes and no arena. Zero
// buffers is a legitimate answer and it is written down, because an absent
// attribute and an attribute of zero are different claims and only one of them
// says the allocator ran.
//
// CHECK-LABEL: func.func @nothing_to_allocate
// CHECK-SAME:    npuisa.fragmentation_ratio = 1.000000e+00 : f64
// CHECK-SAME:    npuisa.scratchpad_bytes = 0 : i64
// CHECK-SAME:    npuisa.scratchpad_peak_bytes = 0 : i64
// CHECK-NEXT:  return
// CHECK-NOT:   memref.alloc
func.func @nothing_to_allocate(%in: memref<1x8x4x4xf32, #npu.dram>) {
  return
}

// =============================================================================
// The statistics, which are what an experiment reads when it wants the numbers
// without parsing the IR. They are summed over every function in the module, so
// these totals are the sum of the per function attributes checked above.
// =============================================================================

// STATS: NPUAllocateScratchpad
// STATS-DAG: allocated-buffers
// STATS-DAG: allocated-bytes
// STATS-DAG: peak-bytes
// STATS-DAG: spilled-buffers
// STATS-DAG: inserted-dma
