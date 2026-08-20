# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The dialect subdirectories arrive with the dialects themselves. At P1 that is
# the npu dialect; the npuisa dialect, the encoding headers and the simulator
# headers join it in their own phases.

add_subdirectory(Dialect)
add_subdirectory(Encoding)
