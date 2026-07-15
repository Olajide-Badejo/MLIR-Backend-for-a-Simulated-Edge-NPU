add_mlir_dialect_library(MLIRNPUDialect
  NPUDialect.cpp
  NPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPU

  DEPENDS
  NPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRTensorDialect
  )
