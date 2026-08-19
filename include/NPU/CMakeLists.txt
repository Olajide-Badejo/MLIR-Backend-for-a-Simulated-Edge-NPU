# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The dialect subdirectories arrive with the dialects themselves: include/NPU
# is empty of TableGen at P0 because P0 ships the skeleton and P1 ships the
# npu dialect. This file exists so the TableGen wiring above it is exercised
# by the P0 configure rather than first exercised at P1.
