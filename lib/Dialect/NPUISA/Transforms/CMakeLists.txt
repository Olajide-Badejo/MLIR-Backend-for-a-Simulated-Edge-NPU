add_mlir_dialect_library(MLIRNPUISATransforms
  LowerNPUToNPUISA.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU/Dialect/NPUISA/Transforms

  DEPENDS
  NPUISAPassesIncGen

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRNPUISADialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
