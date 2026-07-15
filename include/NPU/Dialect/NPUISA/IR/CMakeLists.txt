set(LLVM_TARGET_DEFINITIONS NPUISAOps.td)
mlir_tablegen(NPUISAOps.h.inc -gen-op-decls)
mlir_tablegen(NPUISAOps.cpp.inc -gen-op-defs)
mlir_tablegen(NPUISAOpsDialect.h.inc -gen-dialect-decls -dialect=npuisa)
mlir_tablegen(NPUISAOpsDialect.cpp.inc -gen-dialect-defs -dialect=npuisa)

set(LLVM_TARGET_DEFINITIONS NPUISATypes.td)
mlir_tablegen(NPUISATypes.h.inc -gen-typedef-decls -typedefs-dialect=npuisa)
mlir_tablegen(NPUISATypes.cpp.inc -gen-typedef-defs -typedefs-dialect=npuisa)

add_public_tablegen_target(NPUISAIncGen)
