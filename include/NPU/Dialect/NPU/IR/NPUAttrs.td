//===- NPUAttrs.td - npu dialect attributes ----------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ATTRS
#define NPU_ATTRS

include "NPU/Dialect/NPU/IR/NPUDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Memory space attributes.
//===----------------------------------------------------------------------===//

def NPU_ScratchpadAttr : NPU_Attr<"Scratchpad", "scratchpad"> {
  let summary = "The on chip scratchpad memory space.";
  let description = [{
    Used as the memory space of a `memref` at the `npuisa` level:
    `memref<1x16x8x8xf32, #npu.scratchpad>`.

    This is a zero parameter attribute definition rather than one case of an
    enum attribute, and the difference is visible in the IR: an enum would
    print as `#npu.space<scratchpad>` and every test, every document, and every
    pattern in this project would have to carry the extra word. It prints
    exactly as written above.
  }];
  let assemblyFormat = "";
}

def NPU_DramAttr : NPU_Attr<"Dram", "dram"> {
  let summary = "The off chip DRAM memory space.";
  let description = [{
    The mirror of `#npu.scratchpad`, and the other of the two memory spaces
    this machine has: `memref<1x16x8x8xf32, #npu.dram>`. Function arguments and
    constant data live here, and a DMA is what moves a value between the two.
  }];
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// The layout attribute.
//===----------------------------------------------------------------------===//

def NPU_LayoutKind_NCHW : I32EnumAttrCase<"nchw", 0>;
def NPU_LayoutKind_NHWC : I32EnumAttrCase<"nhwc", 1>;

def NPU_LayoutKind : I32EnumAttr<"LayoutKind",
    "The dimension order of a rank 4 activation tensor",
    [NPU_LayoutKind_NCHW, NPU_LayoutKind_NHWC]> {
  let cppNamespace = "::mlir::npu";
  let genSpecializedAttr = 0;
}

def NPU_LayoutAttr : NPU_Attr<"Layout", "layout"> {
  let summary = "The dimension order of a rank 4 tensor.";
  let description = [{
    Used as the `encoding` of a rank 4 `RankedTensorType`:
    `tensor<1x3x8x8xf32, #npu.layout<nhwc>>`.

    **An absent encoding means NCHW.** That rule is what lets every graph the
    importer emits keep its meaning without an edit, and it is what makes the
    layout assignment pass of a later phase an addition rather than a
    migration. `#npu.layout<nchw>` and no encoding at all are therefore the
    same layout written two ways, and the verifiers treat them as equal.

    The extents themselves are always written in the order the layout names, so
    an NHWC tensor of a batch of 3 channel 8 by 8 images is
    `tensor<1x8x8x3xf32, #npu.layout<nhwc>>`. Reading a shape without reading
    its encoding is therefore always wrong, which is why the shape helpers in
    this dialect take both.

    A layout encoding on a tensor whose rank is not 4 is refused: there is no
    NCHW reading of a rank 2 matrix, and accepting one would mean inventing a
    meaning for it somewhere further down.
  }];

  let parameters = (ins
      EnumParameter<NPU_LayoutKind>:$kind
  );
  let assemblyFormat = "`<` $kind `>`";

  let extraClassDeclaration = [{
    /// True when this layout is NCHW, which is also what an absent encoding
    /// means.
    bool isNCHW() const { return getKind() == ::mlir::npu::LayoutKind::nchw; }
    bool isNHWC() const { return getKind() == ::mlir::npu::LayoutKind::nhwc; }
  }];
}

#endif // NPU_ATTRS
