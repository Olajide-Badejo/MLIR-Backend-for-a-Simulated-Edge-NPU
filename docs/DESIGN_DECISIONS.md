# Design decisions

The choices that shaped the project, and why I made them.

## Builtin tensor type with fixed fp32 elements

The `npu` dialect uses the builtin ranked `tensor` type rather than a custom type.
That keeps interop with the MLIR infrastructure, so the generic folding, dead code
elimination, and conversion machinery apply without op specific glue. Elements are
fixed to fp32, matching the core scope of fp32 CNN inference.

## Bias operand plus activation attribute, not fused op names

Convolution and matmul take an optional bias operand and carry an activation
attribute rather than exploding into a set of named fused ops (conv, conv_bias,
conv_relu, conv_bias_relu, and so on). This mirrors the TFLite style of fused
operators and gives the fusion pass a single clean target: it just sets the
attribute.

## BatchNorm folding as its own named pass

After the weights are loaded, the batch norm scale, offset, mean, and variance are
constants, so the normalization folds into a preceding convolution's weights and
bias by real tensor arithmetic. Making this a dedicated, tested pass keeps the
arithmetic honest and gives the report a concrete subsection.

## No branch instructions in the ISA

Inference graphs are static DAGs, so the instruction stream is straight line. A
branchless ISA is a legitimate simplification, stated as such rather than hidden.

## Two level memory model with explicit DMA

The `npuisa` dialect makes the DRAM and scratchpad split explicit: compute
instructions touch the scratchpad only, and DMA moves data across the boundary.
This is what makes fusion experimentally interesting. A fused conv plus bias plus
relu keeps its intermediate in the scratchpad, and under a constrained budget the
allocator spills fewer buffers, a measurable difference in DRAM traffic that the
benchmark suite reports.

## Dialect conversion for the lowering, not ad hoc rewrites

The tensor to instruction lowering is a representation change (SSA tensors become
scratchpad buffers with addresses). That is exactly what the dialect conversion
framework with a `TypeConverter` and materializations is for, so the lowering uses
it and inserts DMA only at the DRAM boundaries.

## Fixed header plus tagged records for the binary

The `.nbin` format is a fixed header followed by tagged, length prefixed records
rather than a bit packed encoding. It is simpler to get right and to disassemble,
and it is still a real, documented, decodable format.

## Analytical cost model, labeled as such

The simulator's performance numbers come from an analytical model (MACs over
systolic throughput, bytes over DMA bandwidth, elements over lane width, plus a
fixed issue overhead), not a cycle accurate model. Every number it produces is
labeled a simulated estimate, and the constants are documented assumptions.
