add_mlir_dialect_library(MLIRNPUTransforms
  FoldBatchNorm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPU/Transforms

  DEPENDS
  NPUPassesIncGen

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
