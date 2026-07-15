// Scratchpad allocation. Every buffer here is 256 fp32 elements = 1024 bytes.
// Buffer %a is produced first but used only at the very end, so it sits idle
// across the relu chain. With a generous budget everything fits and only
// addresses are assigned. With a budget of 2560 bytes the peak working set (a
// plus two chain buffers = 3072) does not fit, yet no single instruction needs
// more than 2048, so the allocator spills %a to DRAM and reloads it before its
// use.

// RUN: npu-opt %s -npu-allocate-scratchpad | FileCheck %s --check-prefix=FITS
// RUN: npu-opt %s -npu-allocate-scratchpad=budget=2560 | FileCheck %s --check-prefix=SPILL

// FITS: npuisa.scratchpad_budget = 1048576
// FITS-COUNT-2: npuisa.dma_load
// FITS-NOT: npuisa.dma_load
// FITS: address =

// SPILL: npuisa.scratchpad_budget = 2560 : i64
// A spill inserts one extra dma_store after %a and one dma_load reload, so there
// are three of each instead of two.
// SPILL-COUNT-3: npuisa.dma_load
// SPILL-NOT: npuisa.dma_load

func.func @alloc(%xa: tensor<256xf32>, %xb: tensor<256xf32>)
    -> (tensor<256xf32>, tensor<256xf32>) {
  %a = npuisa.dma_load %xa : (tensor<256xf32>) -> !npuisa.buffer<tensor<256xf32>>
  %b = npuisa.dma_load %xb : (tensor<256xf32>) -> !npuisa.buffer<tensor<256xf32>>
  %c = npuisa.relu %b : (!npuisa.buffer<tensor<256xf32>>) -> !npuisa.buffer<tensor<256xf32>>
  %d = npuisa.relu %c : (!npuisa.buffer<tensor<256xf32>>) -> !npuisa.buffer<tensor<256xf32>>
  %e = npuisa.relu %d : (!npuisa.buffer<tensor<256xf32>>) -> !npuisa.buffer<tensor<256xf32>>
  %o1 = npuisa.dma_store %e : (!npuisa.buffer<tensor<256xf32>>) -> tensor<256xf32>
  %f = npuisa.relu %a : (!npuisa.buffer<tensor<256xf32>>) -> !npuisa.buffer<tensor<256xf32>>
  %o2 = npuisa.dma_store %f : (!npuisa.buffer<tensor<256xf32>>) -> tensor<256xf32>
  return %o1, %o2 : tensor<256xf32>, tensor<256xf32>
}
