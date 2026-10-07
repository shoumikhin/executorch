/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_allocator.h>

namespace executorch {
namespace runtime {

DeviceAllocatorRegistry& DeviceAllocatorRegistry::instance() {
  static DeviceAllocatorRegistry registry;
  return registry;
}

} // namespace runtime
} // namespace executorch
