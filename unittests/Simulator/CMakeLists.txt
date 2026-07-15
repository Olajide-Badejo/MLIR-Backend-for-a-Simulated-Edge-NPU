add_llvm_executable(NPUSimulatorTests SimulatorTest.cpp)
llvm_update_compile_flags(NPUSimulatorTests)
target_link_libraries(NPUSimulatorTests PRIVATE
  MLIRNPUSimulator
  MLIRNPUEncoding
  llvm_gtest
  llvm_gtest_main
  )
