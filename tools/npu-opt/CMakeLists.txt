# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The target shape here follows mlir/examples/standalone/standalone-opt/
# CMakeLists.txt from the LLVM project.
#
# MLIRNPU and MLIRNPUTilingInterfaceImpl are named explicitly below rather than
# left to the MLIR_DIALECT_LIBS global property. They do land in that property,
# because add_mlir_dialect_library appends to it, but the property is read here
# before lib/ has been processed on a clean configure, so relying on it would
# make the link line depend on subdirectory order.

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
    MLIRNPU
    MLIRNPUISA
    MLIRNPUISATransforms
    MLIRNPUPipeline
  )

add_llvm_executable(npu-opt npu-opt.cpp)

llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(npu-opt)
