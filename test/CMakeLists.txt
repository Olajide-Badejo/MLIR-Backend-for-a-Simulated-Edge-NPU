# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Derived in shape from mlir/examples/standalone/test/CMakeLists.txt.

llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPU_TEST_DEPENDS
        FileCheck count not
        npu-opt
        )

add_lit_testsuite(check-npu "Running the npu regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${NPU_TEST_DEPENDS}
        )
set_target_properties(check-npu PROPERTIES FOLDER "Tests")

add_lit_testsuites(NPU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NPU_TEST_DEPENDS})
