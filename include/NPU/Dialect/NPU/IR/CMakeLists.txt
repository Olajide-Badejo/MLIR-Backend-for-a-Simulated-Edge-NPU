set(LLVM_TARGET_DEFINITIONS NPUOps.td)
mlir_tablegen(NPUOps.h.inc -gen-op-decls)
mlir_tablegen(NPUOps.cpp.inc -gen-op-defs)
mlir_tablegen(NPUOpsDialect.h.inc -gen-dialect-decls -dialect=npu)
mlir_tablegen(NPUOpsDialect.cpp.inc -gen-dialect-defs -dialect=npu)
mlir_tablegen(NPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(NPUEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(NPUOpsIncGen)

# Regenerate the dialect reference from ODS so docs cannot drift. Run with
# `ninja npu-dialect-doc`. Writes into the source tree docs directory on purpose.
set(_npu_doc_includes
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include)
list(TRANSFORM _npu_doc_includes PREPEND "-I")

add_custom_target(npu-dialect-doc
  COMMAND ${MLIR_TABLEGEN_EXE} -gen-dialect-doc -dialect=npu
          ${_npu_doc_includes}
          ${CMAKE_CURRENT_SOURCE_DIR}/NPUOps.td
          -o ${PROJECT_SOURCE_DIR}/docs/DIALECT_REFERENCE.md
  DEPENDS NPUOpsIncGen
  COMMENT "Regenerating docs/DIALECT_REFERENCE.md from ODS"
  VERBATIM)
