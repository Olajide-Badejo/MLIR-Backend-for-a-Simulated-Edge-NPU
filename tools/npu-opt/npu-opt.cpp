//===- npu-opt.cpp --------------------------------------------*- C++ -*-===//
//
// This file is derived from mlir/examples/standalone/standalone-opt/
// standalone-opt.cpp in the LLVM project and keeps that file's licence.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-FileCopyrightText: Copyright (c) LLVM Project contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The optimizer driver for this project. It registers the upstream MLIR
// dialects and passes, this project's own dialects, and the external interface
// models those dialects promise.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  registry.insert<mlir::npu::NPUDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NPU optimizer driver\n", registry));
}
