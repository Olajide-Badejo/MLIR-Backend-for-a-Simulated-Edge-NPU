# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The library subdirectories arrive with the code they build. At P1 that is the
# npu dialect and its tiling interface models; the npuisa dialect, the encoder,
# the pipeline and the simulator joined in their own phases, and CostModel split
# out of Simulator at P13 when a second caller appeared.
#
# Pipeline is last because it depends on the passes Dialect builds. cmake does
# not require the order, since targets are resolved after the whole tree is
# processed, but a reader looking for what depends on what should not have to
# know that.
#
# CostModel is first for the same reason read the other way: it is the one thing
# here that links nothing, and both of the things that depend on it are below
# it. The simulator charges with it and, from P13,
# `-npu-tile-to-scratchpad` scores its candidate mappings with it. Section 5.5
# requires those to be the same arithmetic rather than two copies of it, and
# this is where that requirement is a build edge rather than a comment.

add_subdirectory(CostModel)
add_subdirectory(Dialect)
add_subdirectory(Encoding)
add_subdirectory(Simulator)
add_subdirectory(Pipeline)
