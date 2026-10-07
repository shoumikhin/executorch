/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/device_allocator.h>

#include <cstdio>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace executorch::runtime;

int main(int argc, char** argv) {
  auto* allocator = get_device_allocator(etensor::DeviceType::CUDA);
  if (allocator == nullptr) {
    std::fprintf(stderr, "CUDA extension did not register its allocator\n");
    return 1;
  }
  for (int i = 1; i < argc; ++i) {
    for (int load = 0; load < 2; ++load) {
      // Keep delegates loaded: the runtime registry retains their addresses.
#if defined(_WIN32)
      if (LoadLibraryA(argv[i]) == nullptr) {
        std::fprintf(
            stderr, "Could not load %s: error %lu\n", argv[i], GetLastError());
        return 1;
      }
#else
      if (dlopen(argv[i], RTLD_NOW | RTLD_LOCAL) == nullptr) {
        std::fprintf(stderr, "Could not load %s: %s\n", argv[i], dlerror());
        return 1;
      }
#endif
      if (get_device_allocator(etensor::DeviceType::CUDA) != allocator) {
        std::fprintf(stderr, "Loading %s changed the CUDA allocator\n", argv[i]);
        return 1;
      }
    }
  }
  return 0;
}
