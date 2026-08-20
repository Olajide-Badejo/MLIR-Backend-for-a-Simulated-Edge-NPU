# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The library subdirectories arrive with the code they build. At P1 that is the
# npu dialect and its tiling interface models; the npuisa dialect, the encoder,
# the pipeline and the simulator join in their own phases.

add_subdirectory(Dialect)
add_subdirectory(Encoding)
