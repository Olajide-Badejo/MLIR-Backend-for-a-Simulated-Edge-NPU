// Phase 0 smoke test: prove npu-opt builds, runs, and round trips a trivial
// builtin module. Real dialect tests arrive in Phase 1.

// RUN: npu-opt %s | FileCheck %s

// CHECK: module
module {
}
