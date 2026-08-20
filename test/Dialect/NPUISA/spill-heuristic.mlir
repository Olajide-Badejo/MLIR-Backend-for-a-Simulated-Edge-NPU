// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Spilling: when it fires, what it emits, and which buffer each heuristic picks.
//
// Four run lines over one file, and the budget is the same on the three that
// spill so that the only variable between them is the heuristic and the
// strategy. A file that changed two things at once could not attribute a
// difference in the output to either.
//
// The fourth run line is the negative case Section 12 requires of every pass: at
// a budget nothing needs spilling for, nothing is spilled. A spiller with only
// positive tests would pass them all while spilling unconditionally, and the
// cost of that would be invisible here and very visible in the DRAM traffic
// numbers three phases later.

// RUN: npu-opt %s \
// RUN:   --npu-allocate-scratchpad="strategy=pack budget=1536 spill-heuristic=longest-range" \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LONGEST
// RUN: npu-opt %s \
// RUN:   --npu-allocate-scratchpad="strategy=pack budget=1536 spill-heuristic=cost" \
// RUN:   | FileCheck %s --check-prefixes=CHECK,COST
// RUN: npu-opt %s --npu-allocate-scratchpad="strategy=interval budget=1536" \
// RUN:   | FileCheck %s --check-prefix=TRIGGER
// RUN: npu-opt %s --npu-allocate-scratchpad="strategy=pack budget=65536" \
// RUN:   | FileCheck %s --check-prefix=NOSPILL

// =============================================================================
// The spill trigger is offset assignment failing, never the peak exceeding the
// budget. Section 13.1 is explicit that getting this backwards spills when it
// need not and fails to spill when it must.
//
// This function is the proof, and it works because the two questions have
// different answers on it:
//
//   %a  512 bytes, live [0, 3]
//   %b  512 bytes, live [1, 5]
//   %c 1024 bytes, live [4, 6]
//
// The sweep line peak is 1536, at index 4 where %b and %c are both live, which
// is exactly the budget. So "peak exceeded budget" is **false** here. The
// interval placement still fails: %a takes 0, %b takes 512 because it overlaps
// %a, and %c cannot start below 1024 because %b is in the way, so it wants
// 1024 to 2048 and the budget stops at 1536.
//
// A spill therefore fires on a program whose peak fits, which no "peak exceeded
// budget" trigger would ever do. After it, %b's live range is cut to its
// definition and its store, the reload lives only across the concatenation, and
// the placement lands at exactly 1536.
//
// The `pack` run lines allocate the same function without spilling at all,
// because greedy by size places %c first and leaves no hole. Same program, same
// budget, same peak, one spill or none depending on the placement: that is the
// difference between the two questions, made visible.
// =============================================================================

// TRIGGER-LABEL: func.func @the_trigger_is_placement_failure_not_the_peak
// TRIGGER-SAME:    npuisa.scratchpad_bytes = 1536 : i64
// TRIGGER-SAME:    npuisa.scratchpad_peak_bytes = 1536 : i64
// TRIGGER-SAME:    npuisa.spill_count = 1 : i64
// TRIGGER-SAME:    npuisa.spill_dma_count = 2 : i64

// CHECK-LABEL:   func.func @the_trigger_is_placement_failure_not_the_peak
// CHECK-SAME:      npuisa.scratchpad_bytes = 1536 : i64
// CHECK-SAME:      npuisa.spill_count = 0 : i64

// NOSPILL-LABEL: func.func @the_trigger_is_placement_failure_not_the_peak
// NOSPILL-SAME:    npuisa.spill_count = 0 : i64
func.func @the_trigger_is_placement_failure_not_the_peak(
    %in: memref<1x8x4x4xf32, #npu.dram>,
    %out: memref<1x16x4x4xf32, #npu.dram>) {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  // TRIGGER: npuisa.relu
  npuisa.relu ins(%a : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
  // The store comes straight after the definition, which is the relu that wrote
  // %b, and the slot is a DRAM buffer marked as what it is.
  // TRIGGER-NEXT: %[[SLOT:.*]] = memref.alloc() {npuisa.spill_slot}
  // TRIGGER-SAME:   : memref<1x8x4x4xf32, #npu.dram>
  // TRIGGER-NEXT: npuisa.dma_store %{{.*}}, %[[SLOT]]
  %c = memref.alloc() : memref<1x16x4x4xf32, #npu.scratchpad>
  // The reload comes before the use, into its own view of the arena, and the
  // concatenation reads the reload rather than the original.
  // TRIGGER: npuisa.dma_load %[[SLOT]], %[[RELOAD:.*]] :
  // TRIGGER-NEXT: npuisa.concat ins(%[[RELOAD]], %[[RELOAD]]
  npuisa.concat ins(%b, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                             memref<1x8x4x4xf32, #npu.scratchpad>)
                outs(%c : memref<1x16x4x4xf32, #npu.scratchpad>) {axis = 1 : i64}
  npuisa.dma_store %c, %out
    : memref<1x16x4x4xf32, #npu.scratchpad> to memref<1x16x4x4xf32, #npu.dram>
  return
}

// =============================================================================
// The two heuristics, on a program where they disagree.
//
// Six buffers. The sweep line peak is 1792 at the allocation of %q, where %big,
// %y, %p and %q are all live, and the budget is 1536, so one of the four gets
// spilled. Their numbers at that index:
//
//   buffer  bytes  span  uses after the peak  cost = bytes * (1 + uses)
//   %big      512    13                    1                      1024
//   %y        256     8                    1                       512
//   %p        512     4                    1                      1024
//   %q        512     5                    1                      1024
//
// `longest-range` spills the longest live range crossing the peak, which is
// %big at 13. `cost` spills the smallest cost, which is %y at 512. Neither is
// the other's answer and neither is a tie, so this file distinguishes the two
// rules rather than merely running both.
//
// The victim is identified by the **type of its spill slot**, which is the one
// thing about a spill that names which buffer it was: %big is 1 x 8 x 4 x 4 and
// %y is 1 x 4 x 4 x 4. Checking an offset instead would be checking the
// placement that followed the decision rather than the decision.
// =============================================================================

// CHECK-LABEL: func.func @two_victims
// CHECK-SAME:    npuisa.spill_count = 1 : i64
// CHECK-SAME:    npuisa.spill_dma_count = 2 : i64

// LONGEST: memref.alloc() {npuisa.spill_slot} : memref<1x8x4x4xf32, #npu.dram>
// LONGEST-NOT: memref.alloc() {npuisa.spill_slot} : memref<1x4x4x4xf32, #npu.dram>

// COST: memref.alloc() {npuisa.spill_slot} : memref<1x4x4x4xf32, #npu.dram>
// COST-NOT: memref.alloc() {npuisa.spill_slot} : memref<1x8x4x4xf32, #npu.dram>

// At a budget nothing needs spilling for, nothing is spilled and no DMA is
// added. This is the negative case, and it is the one that would catch a
// spiller that fired on every function.
//
// NOSPILL-LABEL: func.func @two_victims
// NOSPILL-SAME:    npuisa.scratchpad_bytes = 1792 : i64
// NOSPILL-SAME:    npuisa.scratchpad_peak_bytes = 1792 : i64
// NOSPILL-SAME:    npuisa.spill_count = 0 : i64
// NOSPILL-SAME:    npuisa.spill_dma_count = 0 : i64
// NOSPILL-NOT:     npuisa.spill_slot
func.func @two_victims(%in0: memref<1x8x4x4xf32, #npu.dram>,
                       %in1: memref<1x4x4x4xf32, #npu.dram>,
                       %in2: memref<1x8x4x4xf32, #npu.dram>,
                       %o0: memref<1x8x4x4xf32, #npu.dram>,
                       %o1: memref<1x4x4x4xf32, #npu.dram>,
                       %o2: memref<1x8x4x4xf32, #npu.dram>) {
  %big = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in0, %big
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %y = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in1, %y
    : memref<1x4x4x4xf32, #npu.dram> to memref<1x4x4x4xf32, #npu.scratchpad>
  %p = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in2, %p
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %q = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%p : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%q : memref<1x8x4x4xf32, #npu.scratchpad>)
  %yy = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%y : memref<1x4x4x4xf32, #npu.scratchpad>)
              outs(%yy : memref<1x4x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %q, %o0
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  npuisa.dma_store %yy, %o1
    : memref<1x4x4x4xf32, #npu.scratchpad> to memref<1x4x4x4xf32, #npu.dram>
  npuisa.dma_store %big, %o2
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}
