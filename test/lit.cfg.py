# -*- Python -*-
#
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Derived in shape from mlir/examples/standalone/test/lit.cfg.py.

import os

import lit.formats
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

config.name = "NPU"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.npu_obj_root, "test")

config.npu_tools_dir = os.path.join(config.npu_obj_root, "bin")
config.npu_libs_dir = os.path.join(config.npu_obj_root, "lib")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%npu_libs", config.npu_libs_dir))
config.substitutions.append(("%npu_src_root", config.npu_src_root))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# Inputs directories hold auxiliary files for the tests beside them and are not
# tests themselves.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.npu_tools_dir, config.llvm_tools_dir]
tools = [
    "mlir-opt",
    "npu-opt",
    "npu-translate",
    "npu-objdump",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# The MLIR Python bindings resolve only out of the LLVM build tree; they are
# not pip installed and lit does not inherit the developer's shell. Section 3.3
# of the build specification names this one of the three places the wiring has
# to exist, the others being pyproject.toml and the pytest configuration. A
# missing entry here surfaces as `ModuleNotFoundError: mlir` inside a test that
# has nothing to do with the bindings, which is why it is set explicitly rather
# than left to the environment.
python_path = [config.mlir_python_packages_dir]
if "PYTHONPATH" in os.environ:
    python_path += [os.environ["PYTHONPATH"]]

llvm_config.with_environment("PYTHONPATH", python_path, append_path=True)
