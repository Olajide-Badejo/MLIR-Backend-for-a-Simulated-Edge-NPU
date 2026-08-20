# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The generator of Section 9.4. It is a TableGen driver rather than a script,
# because the project already depends on TableGen and already has a staleness
# gate shaped around TableGen generation, and a second mechanism would be a
# second thing to keep working.
#
# LLVM_LINK_COMPONENTS is read by add_llvm_executable from this scope. TableGen
# is the component that carries the record parser and TableGenMain; it is
# exported by LLVMConfig.cmake from a build tree and from an install prefix
# alike, which is what lets this tool build in CI against /opt/llvm.

set(LLVM_LINK_COMPONENTS
        Support
        TableGen
        )

add_llvm_executable(npu-isa-tblgen npu-isa-tblgen.cpp)

llvm_update_compile_flags(npu-isa-tblgen)
