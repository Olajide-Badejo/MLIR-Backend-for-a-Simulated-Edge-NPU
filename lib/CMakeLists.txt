# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The library subdirectories arrive with the code they build. At P1 that is the
# npu dialect and its tiling interface models; the npuisa dialect, the encoder,
# the pipeline and the simulator joined in their own phases, and with Pipeline
# at P8 the list is complete.
#
# Pipeline is last because it depends on the passes Dialect builds. cmake does
# not require the order, since targets are resolved after the whole tree is
# processed, but a reader looking for what depends on what should not have to
# know that.

add_subdirectory(Dialect)
add_subdirectory(Encoding)
add_subdirectory(Simulator)
add_subdirectory(Pipeline)
