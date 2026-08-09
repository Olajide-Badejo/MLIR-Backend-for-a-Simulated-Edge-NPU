add_llvm_executable(NPUEncodingTests
  EncodingTest.cpp
  ValidationTest.cpp
  PropertyTest.cpp
  FuzzTest.cpp
  )
llvm_update_compile_flags(NPUEncodingTests)
target_link_libraries(NPUEncodingTests PRIVATE
  MLIRNPUEncoding
  MLIRNPUSimulator
  MLIRNPUISADialect
  MLIRParser
  MLIRIR
  llvm_gtest
  llvm_gtest_main
  )
