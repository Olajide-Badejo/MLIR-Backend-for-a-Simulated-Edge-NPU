import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the lit test runner.

config.name = "NPU"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.npu_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]

tool_dirs = [config.npu_tools_dir, config.llvm_tools_dir]
tools = ["npu-opt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
