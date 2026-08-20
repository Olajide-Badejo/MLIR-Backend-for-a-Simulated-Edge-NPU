//===- NPUISADescription.td - the one source for the ISA ---*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The instruction set of Section 5.4, described once.
//
// Before this file existed the semantics of the instruction set lived in four
// hand maintained places: docs/ISA_MANUAL.md, the opcode table in Program.h,
// the arity and shape rules in Program::validate(), and the kernels in the
// simulator. Nothing but discipline kept them consistent, and this project's
// own position is that discipline is not a mechanism. The dialect reference
// already solves the same problem the same way, by generating itself from ODS
// with a staleness gate in CI, so the ISA is solved the same way rather than a
// second way.
//
// What `npu-isa-tblgen` generates from this file:
//
//   -gen-opcode-enum      the Opcode enum, kMaxOpcode, and the name table
//   -gen-opcode-info      the arity, field presence, memory space and element
//                         type rules that Program::validate() reads
//   -gen-check-enum       the validation check names of Section 9.2
//   -gen-dispatch-def     the simulator's dispatch skeleton, as an X macro
//   -gen-manual           the generated sections of docs/ISA_MANUAL.md
//   -gen-json             the opcode list scripts/check-reachability.py reads
//
// **What is deliberately not generated: the kernels.** The arithmetic is the
// interesting part and a generated convolution would be a worse convolution.
// What is generated is the surface that is mechanical and therefore drifts.
//
// THE RULE THAT GOVERNS EDITS TO THIS FILE. Opcode numeric values are assigned
// once and are never renumbered. New opcodes are appended with the next free
// value. Any change to the *layout* of the binary format bumps
// `Program::kVersion`; adding an opcode does not, because an opcode value is
// data inside a layout that did not move.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ISA_DESCRIPTION
#define NPU_ISA_DESCRIPTION

//===----------------------------------------------------------------------===//
// The vocabulary.
//===----------------------------------------------------------------------===//

// An element type, as it appears in the binary. The numeric values are the ones
// Section 9.1 fixes from the first version of the format, and they are here
// rather than only in C++ so that the manual and the validator read them from
// the same place.
class ISAElemType<int val, string spelling> {
  int value = val;
  string name = spelling;
}

def F32 : ISAElemType<0, "f32">;
def I8  : ISAElemType<1, "i8">;
def I32 : ISAElemType<2, "i32">;

// A memory space. The compute units of this machine address the scratchpad and
// nothing else; DRAM is reachable only through the two DMA instructions.
class ISAMemSpace<int val, string spelling> {
  int value = val;
  string name = spelling;
}

def Scratchpad : ISAMemSpace<0, "scratchpad">;
def Dram       : ISAMemSpace<1, "dram">;

// A field of the `Instruction` record that an opcode gives meaning to. Every
// field is physically present on every instruction, because the format has no
// tags and therefore no optional fields; what varies is which ones carry
// meaning, and an opcode that gives a field no meaning requires it to hold its
// neutral value. That is what makes `attribute-size` and `attribute-value`
// checkable rather than advisory.
class ISAField<string spelling, string doc> {
  string name = spelling;
  string description = doc;
}

def FieldPads : ISAField<"pads",
    "four entries in ONNX order: padTop, padLeft, padBottom, padRight">;
def FieldStrides : ISAField<"strides",
    "two entries, the window stride, height then width">;
def FieldDilations : ISAField<"dilations",
    "two entries, the window dilation, height then width">;
def FieldKernel : ISAField<"kernel",
    "two entries, the window extent, height then width">;
def FieldGroup : ISAField<"group",
    "the number of channel groups, which divides both channel counts">;
def FieldAxes : ISAField<"axes",
    "the permutation for TRANSPOSE, the single concatenation axis for CONCAT">;
def FieldScale : ISAField<"scale",
    "the quantization scale, a finite positive f32">;
def FieldZeroPoint : ISAField<"zeroPoint",
    "the quantization zero point, within the integer type's range">;
def FieldRequantize : ISAField<"requantize",
    "requantMultiplier and requantShift, the fixed point rescaling pair">;
def FieldActivation : ISAField<"activation",
    "a fused activation applied to the result">;

// An activation value. `none` is the neutral value that every opcode which
// does not list FieldActivation is required to carry, so that a file setting an
// activation on an opcode with no activation unit is a rejected file rather
// than a field somebody ignores.
class ISAActivation<int val, string spelling> {
  int value = val;
  string name = spelling;
}

def ActNone : ISAActivation<0, "none">;
def ActRelu : ISAActivation<1, "relu">;

//===----------------------------------------------------------------------===//
// The opcode.
//===----------------------------------------------------------------------===//

// One instruction of the machine.
//
// `minOperands` and `maxOperands` bound the operand count; a `maxOperands` of
// -1 means variadic with no upper bound beyond `kMaxCount`. `operandSpaces`
// gives the memory space of each operand slot, and its last entry repeats for
// every operand a variadic opcode has beyond the ones listed.
class ISAOpcode<int val, string sem> {
  // The numeric value in the binary. Assigned once, never renumbered.
  int value = val;
  // One line of semantics for the manual.
  string semantics = sem;

  int minOperands = 0;
  int maxOperands = 0;

  // Whether the instruction writes a result buffer. NOP and HALT do not, and
  // their result fields are required to be empty.
  bit hasResult = 1;

  list<ISAMemSpace> operandSpaces = [];
  ISAMemSpace resultSpace = Scratchpad;

  // The fields this opcode gives meaning to. Every other field must hold its
  // neutral value.
  list<ISAField> fields = [];

  // The element types this opcode accepts on its result. An opcode whose
  // operands take a different type from the result says so in `operandTypes`;
  // an empty `operandTypes` means the operands take the result's type.
  list<ISAElemType> elementTypes = [F32];
  list<ISAElemType> operandTypes = [];

  // The shape relation, for the manual. Prose, because the mechanical half is
  // generated and the rest is genuinely semantic.
  string shapeRule = "";

  // The disassembler's format string. `%r` is the result, `%0`, `%1`, `%2` are
  // operands, `%n` is the operand list of a variadic opcode, and a field is
  // named in braces.
  string format = "";

  // The operations of the `npu` dialect that reach this opcode. Read by
  // scripts/check-reachability.py as the encoding layer of law 2, so that the
  // layer is decided from this description rather than by grepping C++.
  list<string> sources = [];

  // Whether the simulator needs a kernel for this opcode. NOP and HALT are
  // control, not computation, and the dispatch skeleton marks them so.
  bit needsKernel = 1;

  // The phase at which the semantics land. Structural coverage exists from P6
  // for every opcode; QUANT and DEQUANT get their integer kernels at P14, and
  // saying so here keeps the manual honest about what is implemented.
  string semanticPhase = "P7";
}

//===----------------------------------------------------------------------===//
// The instruction set of Section 5.4, in its fixed numbering.
//===----------------------------------------------------------------------===//

def NOP : ISAOpcode<0, "Does nothing and advances the program counter."> {
  let hasResult = 0;
  let needsKernel = 0;
  let format = "NOP";
  let shapeRule = "No operands and no result.";
  let semanticPhase = "P6";
}

def HALT : ISAOpcode<1, "Stops the machine. The encoder emits one as the last instruction of every program."> {
  let hasResult = 0;
  let needsKernel = 0;
  let format = "HALT";
  let shapeRule = "No operands and no result.";
  let semanticPhase = "P6";
}

def DMA_LOAD : ISAOpcode<2, "Copies a buffer from DRAM into the scratchpad."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Dram];
  let resultSpace = Scratchpad;
  let elementTypes = [F32, I8, I32];
  let format = "DMA_LOAD %r <- %0";
  let shapeRule = "The operand and the result have the same shape and the same "
                  "element type. The DMA moves bytes and does not change a "
                  "layout on the way.";
  let sources = [];
}

def DMA_STORE : ISAOpcode<3, "Copies a buffer from the scratchpad back to DRAM."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let resultSpace = Dram;
  let elementTypes = [F32, I8, I32];
  let format = "DMA_STORE %r <- %0";
  let shapeRule = "The mirror of DMA_LOAD: same shape, same element type.";
  let sources = [];
}

def MATMUL : ISAOpcode<4, "A rank 2 by rank 2 matrix multiplication with an optional bias."> {
  let minOperands = 2;
  let maxOperands = 3;
  let operandSpaces = [Scratchpad, Scratchpad, Scratchpad];
  let fields = [FieldRequantize, FieldActivation];
  let format = "MATMUL %r <- %0, %1 {bias %2} {activation}";
  let shapeRule = "Operand 0 is (M, K), operand 1 is (K, N), the result is "
                  "(M, N). The optional operand 2 is the bias and has length N.";
  let sources = ["matmul"];
}

def CONV2D : ISAOpcode<5, "A two dimensional grouped convolution with an optional bias."> {
  let minOperands = 2;
  let maxOperands = 3;
  let operandSpaces = [Scratchpad, Scratchpad, Scratchpad];
  let fields = [FieldStrides, FieldPads, FieldDilations, FieldGroup,
                FieldRequantize, FieldActivation];
  let format = "CONV2D %r <- %0, %1 {bias %2} {strides} {pads} {dilations} {group} {activation}";
  let shapeRule = "Operand 0 is (N, C, H, W), operand 1 is (F, C/group, KH, KW), "
                  "the result is (N, F, OH, OW). The optional operand 2 is the "
                  "bias and has length F.";
  let sources = ["conv2d"];
}

def ADD : ISAOpcode<6, "Elementwise addition."> {
  let minOperands = 2;
  let maxOperands = 2;
  let operandSpaces = [Scratchpad, Scratchpad];
  let fields = [FieldRequantize, FieldActivation];
  let format = "ADD %r <- %0, %1 {activation}";
  let shapeRule = "Both operands have the result's shape. No broadcasting: the "
                  "frontend materialises every broadcast it accepts, and a rank "
                  "1 channel operand arrives as a stride 0 operand rather than "
                  "as a shape.";
  let sources = ["add", "batch_norm"];
}

def MUL : ISAOpcode<7, "Elementwise multiplication."> {
  let minOperands = 2;
  let maxOperands = 2;
  let operandSpaces = [Scratchpad, Scratchpad];
  let fields = [FieldRequantize, FieldActivation];
  let format = "MUL %r <- %0, %1 {activation}";
  let shapeRule = "The same rule as ADD.";
  let sources = ["mul", "batch_norm"];
}

def RELU : ISAOpcode<8, "The rectified linear unit, elementwise."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let format = "RELU %r <- %0";
  let shapeRule = "The operand has the result's shape. The operand and the "
                  "result may be the same buffer: an in place relu is what the "
                  "allocator produces when it reuses a dead interval.";
  let sources = ["relu"];
}

def POOL_MAX : ISAOpcode<9, "Two dimensional max pooling."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let fields = [FieldKernel, FieldStrides, FieldPads, FieldDilations];
  let format = "POOL_MAX %r <- %0 {kernel} {strides} {pads} {dilations}";
  let shapeRule = "Operand and result are both (N, C, H, W) with the same N and "
                  "C. The output extents are the result's, taken as declared "
                  "rather than recomputed, because the encoder has already "
                  "resolved ceil_mode.";
  let sources = ["max_pool2d"];
}

def POOL_AVG : ISAOpcode<10, "Two dimensional average pooling."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let fields = [FieldKernel, FieldStrides, FieldPads, FieldDilations];
  let format = "POOL_AVG %r <- %0 {kernel} {strides} {pads} {dilations}";
  let shapeRule = "The same rule as POOL_MAX. The mean divides by the number of "
                  "elements that actually contributed, which is ONNX's "
                  "count_include_pad = 0 and the only behaviour implemented.";
  let sources = ["avg_pool2d"];
}

def RESHAPE : ISAOpcode<11, "Copies a buffer under a different set of extents."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let elementTypes = [F32, I8, I32];
  let format = "RESHAPE %r <- %0";
  let shapeRule = "Element counts and element types agree; the extents need "
                  "not.";
  let sources = ["reshape"];
}

def TRANSPOSE : ISAOpcode<12, "Permutes a buffer's dimensions into the result."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let fields = [FieldAxes];
  let elementTypes = [F32, I8, I32];
  let format = "TRANSPOSE %r <- %0 {axes}";
  let shapeRule = "`axes` is a permutation of exactly the result rank, and "
                  "result extent i equals operand extent axes[i].";
  let sources = ["transpose"];
}

def CONCAT : ISAOpcode<13, "Concatenates buffers along one axis into the result."> {
  let minOperands = 1;
  let maxOperands = -1;
  let operandSpaces = [Scratchpad];
  let fields = [FieldAxes];
  let elementTypes = [F32, I8, I32];
  let format = "CONCAT %r <- %n {axes}";
  let shapeRule = "`axes` holds exactly one entry, the axis, non negative and "
                  "less than the rank. Every operand has the result's rank, the "
                  "extents agree on every other axis, and along the axis they "
                  "sum to the result extent.";
  let sources = ["concat"];
}

def QUANT : ISAOpcode<14, "Quantizes f32 to i8 with a scale and a zero point."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let fields = [FieldScale, FieldZeroPoint];
  let elementTypes = [I8];
  let operandTypes = [F32];
  let format = "QUANT %r <- %0 {scale} {zeroPoint}";
  let shapeRule = "The operand and the result have the same shape. The operand "
                  "is f32 and the result is i8.";
  let sources = [];
  let semanticPhase = "P14";
}

def DEQUANT : ISAOpcode<15, "Dequantizes i8 to f32 with a scale and a zero point."> {
  let minOperands = 1;
  let maxOperands = 1;
  let operandSpaces = [Scratchpad];
  let fields = [FieldScale, FieldZeroPoint];
  let elementTypes = [F32];
  let operandTypes = [I8];
  let format = "DEQUANT %r <- %0 {scale} {zeroPoint}";
  let shapeRule = "The mirror of QUANT: same shape, i8 operand, f32 result.";
  let sources = [];
  let semanticPhase = "P14";
}

//===----------------------------------------------------------------------===//
// The `npu` operations that reach the encoder by elimination.
//
// Law 2 asks that every operation of the `npu` dialect be encodable. Two of
// them are encodable by being gone: they are structural, they describe how the
// graph is written rather than what the machine does, and the lowering removes
// them before any instruction is emitted. Recording that here rather than in a
// comment in the encoder is what lets the reachability checker decide the
// encoding layer from this description instead of by grepping a source file.
//===----------------------------------------------------------------------===//

class ISAEliminatedSource<string op, string how> {
  string source = op;
  string explanation = how;
}

def : ISAEliminatedSource<"fused_op",
    "flattened into its parent block by -npu-lower-to-npuisa before any "
    "instruction is emitted, so the operations the region held are encoded and "
    "the region is not">;

def : ISAEliminatedSource<"yield",
    "the terminator of an npu.fused_op region, which disappears with the "
    "region when the fused operation is flattened">;

def : ISAEliminatedSource<"constant",
    "encoded as a constant region in the DRAM map with its data, not as an "
    "instruction; the load that brings it on chip is a DMA_LOAD">;

//===----------------------------------------------------------------------===//
// The validation checks of Section 9.2.
//
// The names are here rather than only in Validation.cpp so that the manual
// takes them from the source instead of copying them, which is the workaround
// Section 9.2 already asked for and which this file turns into a mechanism. A
// check that exists in C++ and not here fails to compile, because the enum it
// switches on is generated from this list.
//===----------------------------------------------------------------------===//

class ISACheck<int ord, string spelling, string doc> {
  int order = ord;
  string name = spelling;
  string description = doc;
}

def CheckVersion : ISACheck<0, "version",
    "The version word is one this decoder accepts. An unknown version is "
    "rejected by name rather than reinterpreted.">;
def CheckStructure : ISACheck<1, "structure",
    "The file is framed correctly: the magic matches, every section is fully "
    "present, and nothing follows the last one.">;
def CheckArity : ISACheck<2, "arity",
    "The operand count is within the range the opcode declares.">;
def CheckCountCap : ISACheck<3, "count-cap",
    "Every u32 count field is at most kMaxCount. The cap is a number and it is "
    "checked before anything is allocated.">;
def CheckResultShape : ISACheck<4, "result-shape",
    "Every result extent is positive and the product stays within the shape "
    "limit, tested without ever performing the multiplication that would "
    "overflow. It also carries the one shape relation an opcode gets to choose "
    "for itself: a RESHAPE's result holds exactly as many elements as its "
    "operand.">;
def CheckResultAddress : ISACheck<5, "result-address",
    "The result address is non negative, is zero when the opcode writes no "
    "result, and names the memory space the opcode writes its result in.">;
def CheckResultInRange : ISACheck<6, "result-in-range",
    "The result's bytes lie inside the scratchpad the file declares.">;
def CheckOperandInRange : ISACheck<7, "operand-in-range",
    "Each scratchpad operand's byte span lies inside the scratchpad the file "
    "declares, and every operand names the memory space the opcode declares "
    "for that slot.">;
def CheckOperandDefined : ISACheck<8, "operand-defined",
    "Each operand's bytes were written by an earlier instruction, or belong to "
    "a declared input or constant region. An output region and a spill slot "
    "are places to write, so reading one before writing it is reading whatever "
    "the loader left there.">;
def CheckOperandExtent : ISACheck<9, "operand-extent",
    "The consumer's element need fits the element count actually written to "
    "the buffer it reads. Membership alone is not enough: a DMA_STORE reading "
    "100 elements from a 4 element buffer would pass a membership test and "
    "then trap.">;
def CheckDramAddress : ISACheck<10, "dram-address",
    "A DRAM address is non negative.">;
def CheckDramInRange : ISACheck<11, "dram-in-range",
    "A DRAM access lies inside the declared DRAM size.">;
def CheckRegionOffset : ISACheck<12, "region-offset",
    "A memory region's DRAM offset is non negative and does not overflow when "
    "its size is added.">;
def CheckRegionShape : ISACheck<13, "region-shape",
    "A memory region's extents are positive and its element count stays within "
    "the shape limit.">;
def CheckRegionInRange : ISACheck<14, "region-in-range",
    "A memory region lies inside the declared DRAM size.">;
def CheckConstantData : ISACheck<15, "constant-data",
    "A constant's data length equals its region's byte size, computed from the "
    "element type rather than assumed to be four bytes per element.">;
def CheckAttributeSize : ISACheck<16, "attribute-size",
    "Each attribute vector has the length the opcode declares, and an "
    "attribute the opcode gives no meaning to is empty.">;
def CheckAttributeValue : ISACheck<17, "attribute-value",
    "Strides and dilations are positive, pads are non negative, and the group "
    "count is positive.">;
def CheckActivation : ISACheck<18, "activation",
    "The activation field holds a defined value, and holds `none` on an opcode "
    "that fuses no activation.">;
def CheckElementType : ISACheck<19, "element-type",
    "Every element type byte is one of the defined values.">;
def CheckElementTypeSupported : ISACheck<20, "element-type-supported",
    "The opcode accepts the element type it was given.">;
def CheckQuantScale : ISACheck<21, "quant-scale",
    "The quantization scale is finite and strictly positive, and is zero on an "
    "opcode that does not quantize.">;
def CheckQuantZeroPoint : ISACheck<22, "quant-zero-point",
    "The zero point lies inside the range of the integer type it belongs to.">;
def CheckQuantTypes : ISACheck<23, "quant-types",
    "QUANT reads f32 and writes i8; DEQUANT reads i8 and writes f32.">;
def CheckQuantShape : ISACheck<24, "quant-shape",
    "A quantization instruction's operand and result have the same shape.">;
def CheckQuantRequantize : ISACheck<25, "quant-requantize",
    "The requantization shift is within [0, 31] and the multiplier is a "
    "positive int32, which is the range the fixed point decomposition "
    "produces. A value outside it is a corrupt file, not an exotic "
    "configuration.">;
def CheckAxesPermutation : ISACheck<26, "axes-permutation",
    "TRANSPOSE's axes vector is a permutation of exactly the result rank.">;
def CheckConcatAxis : ISACheck<27, "concat-axis",
    "CONCAT's axes vector holds one entry, non negative and less than the "
    "rank.">;
def CheckConcatExtents : ISACheck<28, "concat-extents",
    "CONCAT's operands agree on every axis but the concatenation axis and sum "
    "to the result extent along it.">;
def CheckDebugPc : ISACheck<29, "debug-pc",
    "A debug entry's program counter names an instruction that exists.">;
def CheckDebugOrder : ISACheck<30, "debug-order",
    "Debug entries are strictly increasing in program counter, so a lookup is "
    "a binary search and a duplicate is a corrupt file.">;
def CheckDebugName : ISACheck<31, "debug-name",
    "A debug name holds no NUL and no byte above 0x7f. ONNX node names are "
    "ASCII, and a name that is not is a file somebody built by hand.">;
def CheckDebugSize : ISACheck<32, "debug-size",
    "A debug name's length is within kMaxCount and within the bytes the file "
    "actually has.">;

#endif // NPU_ISA_DESCRIPTION
