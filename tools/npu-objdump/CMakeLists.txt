add_llvm_executable(npu-objdump npu-objdump.cpp)
llvm_update_compile_flags(npu-objdump)
target_link_libraries(npu-objdump PRIVATE MLIRNPUEncoding)
