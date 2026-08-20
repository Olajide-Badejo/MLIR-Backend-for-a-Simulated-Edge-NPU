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
// The tiling interface registration is a separate call rather than something
// the dialect does for itself, and that is the whole point of the promised
// interface mechanism: the dialect library carries no dependency on the tiling
// stack, and a tool that forgets this line gets a named error at the first use
// saying the interface was promised and never provided.
//
// This project's own passes are registered by their generated entry point,
// beside MLIR's. No pipeline is assembled here: the driver's `-O` levels belong
// to the phase that has a whole pipeline to describe, and a pipeline name
// registered before there is anything to put behind it would promise more than
// it delivered.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/Interfaces/NPUTilingInterfaceImpl.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::npuisa::registerNPUISAPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  registry.insert<mlir::npu::NPUDialect, mlir::npuisa::NPUISADialect>();
  mlir::npu::registerNPUTilingInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NPU optimizer driver\n", registry));
}
