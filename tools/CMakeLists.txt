# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT

# npu-isa-tblgen is not here. The top level CMakeLists adds it before
# include/, because the generation in include/NPU/Encoding runs the tool.

add_subdirectory(npu-opt)
add_subdirectory(npu-translate)
add_subdirectory(npu-objdump)
add_subdirectory(npu-sim)
