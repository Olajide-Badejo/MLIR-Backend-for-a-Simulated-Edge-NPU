add_llvm_executable(NPUEncodingTests EncodingTest.cpp)
llvm_update_compile_flags(NPUEncodingTests)
target_link_libraries(NPUEncodingTests PRIVATE
  MLIRNPUEncoding
  MLIRNPUISADialect
  MLIRParser
  MLIRIR
  llvm_gtest
  llvm_gtest_main
  )
