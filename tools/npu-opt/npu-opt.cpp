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
// The optimizer driver for this project. At P0 it registers upstream MLIR
// dialects and passes and nothing else, because the npu dialect does not
// exist until P1. That is deliberate: it proves the out of tree build, the
// link line and the lit substitution are all correct before there is any
// dialect code that could be blamed for a failure in them.
//
//===----------------------------------------------------------------------===//

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

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NPU optimizer driver\n", registry));
}
