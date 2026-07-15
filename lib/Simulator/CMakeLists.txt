add_mlir_library(MLIRNPUSimulator
  Simulator.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU/Simulator

  LINK_LIBS PUBLIC
  MLIRNPUEncoding
  )
