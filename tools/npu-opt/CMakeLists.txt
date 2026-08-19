# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The target shape here follows mlir/examples/standalone/standalone-opt/
# CMakeLists.txt from the LLVM project. The library list is shorter because
# at P0 there is no project dialect to link in.

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
  )

add_llvm_executable(npu-opt npu-opt.cpp)

llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(npu-opt)
