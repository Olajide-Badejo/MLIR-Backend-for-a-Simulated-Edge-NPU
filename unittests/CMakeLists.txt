# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The GoogleTest binaries. Each one arrives with the code it tests, so this file
# grows a subdirectory per phase rather than listing targets that do not exist.
#
# add_mlir_unittest comes from AddMLIR and puts the binary in build/bin, which
# is where the CI activation table's guards look for it.

add_subdirectory(Dialect)
add_subdirectory(Encoding)
add_subdirectory(Simulator)
