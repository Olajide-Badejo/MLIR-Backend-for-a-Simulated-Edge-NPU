get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  # The registerAllDialects/registerAllPasses/registerAllExtensions entry points
  # live in their own aggregator libraries in this LLVM release; link them so the
  # symbols the driver calls are defined.
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
  MLIROptLib
  MLIRPass
)

add_llvm_executable(npu-opt npu-opt.cpp)
llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(npu-opt)
