# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The TableGen wiring for the npu dialect, following the shape of
# mlir/examples/standalone/include/Standalone/CMakeLists.txt.

# add_mlir_dialect generates the op, dialect, type and attribute declarations
# and definitions from NPUOps.td and creates the MLIRNPUOpsIncGen target that
# the library DEPENDS on.
add_mlir_dialect(NPUOps npu)

# add_mlir_dialect generates op, dialect and type declarations, and nothing
# else. The attribute definitions and the enum backing #npu.layout need their
# own passes over the same source file, added by hand here. Leaving them out is
# not a build error, it is a link error at the first use of the attribute, so
# the omission is worth being explicit about.
set(LLVM_TARGET_DEFINITIONS NPUOps.td)
mlir_tablegen(NPUOpsAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=npu)
mlir_tablegen(NPUOpsAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=npu)
mlir_tablegen(NPUOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(NPUOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRNPUOpsAttrsIncGen)

# The dialect reference of the documentation set, regenerated as a build target
# so it cannot drift from the source. add_mlir_doc writes into
# ${CMAKE_BINARY_DIR}/docs/, and the npu-dialect-doc target defined at the top
# level copies the result over docs/DIALECT_REFERENCE.md; CI regenerates and
# diffs. Naming the output NPUDialectReference keeps the generated file
# distinct from the committed one, so a stale committed file is a diff rather
# than a file that quietly overwrote itself.
add_mlir_doc(NPUOps NPUDialectReference NPU/ -gen-dialect-doc -dialect=npu)
