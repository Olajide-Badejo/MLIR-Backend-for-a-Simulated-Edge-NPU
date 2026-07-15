add_llvm_executable(npu-translate npu-translate.cpp)
llvm_update_compile_flags(npu-translate)
target_link_libraries(npu-translate PRIVATE
  MLIRNPUEncoding
  MLIRNPUDialect
  MLIRNPUISADialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRParser
  MLIRIR
  )
