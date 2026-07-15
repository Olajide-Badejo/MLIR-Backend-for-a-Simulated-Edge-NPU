add_mlir_library(MLIRNPUEncoding
  Program.cpp
  Disassembler.cpp
  InstructionEncoder.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU/Encoding

  LINK_LIBS PUBLIC
  MLIRNPUISADialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  )
