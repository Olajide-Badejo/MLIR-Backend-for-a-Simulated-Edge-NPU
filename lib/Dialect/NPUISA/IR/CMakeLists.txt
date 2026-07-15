add_mlir_dialect_library(MLIRNPUISADialect
  NPUISADialect.cpp
  NPUISAOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPUISA

  DEPENDS
  NPUISAIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTensorDialect
  )
