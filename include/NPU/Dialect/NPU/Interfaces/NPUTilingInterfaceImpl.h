//===- NPUTilingInterfaceImpl.h - TilingInterface models --------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// TilingInterface is implemented on the `npu` tensor operations, not on the
// `npuisa` memref ones, and the reason is in the pass order: the tiling pass
// runs before lowering, so the interface has to exist where that pass can see
// it. `getResultTilePosition` and `generateResultTileValue` are the methods
// that make tile and fuse work, and they are meaningful only on operations that
// have results. Putting the interface on the memref side would buy iteration
// domain introspection and nothing else, and would leave tile and fuse
// unreachable.
//
// The models are registered as external models from this separate translation
// unit so the dialect library itself keeps no dependency on the SCF and tensor
// tiling stack. `NPUDialect::initialize` promises the interface with
// `declarePromisedInterface`, so a caller who forgets to call the registration
// function below gets a named error saying the interface was promised and never
// provided, instead of a silent report that the operation does not implement
// TilingInterface.
//
// The interface is implemented here and CONSUMED in a later phase, on purpose.
// Implementing an interface and using it in the same session makes it
// impossible to tell an interface bug from a policy bug.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPU_INTERFACES_NPUTILINGINTERFACEIMPL_H
#define NPU_DIALECT_NPU_INTERFACES_NPUTILINGINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace npu {

/// Attaches the TilingInterface external models to the npu compute operations.
/// Every tool that may run a tiling pass calls this; `npu-opt` calls it at
/// startup.
void registerNPUTilingInterfaceExternalModels(DialectRegistry &registry);

} // namespace npu
} // namespace mlir

#endif // NPU_DIALECT_NPU_INTERFACES_NPUTILINGINTERFACEIMPL_H
