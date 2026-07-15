add_llvm_executable(npu-sim npu-sim.cpp)
llvm_update_compile_flags(npu-sim)
target_link_libraries(npu-sim PRIVATE MLIRNPUSimulator MLIRNPUEncoding)
