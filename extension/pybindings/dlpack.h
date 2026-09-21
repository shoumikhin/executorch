/*
 * Copyright (c) 2017 by Contributors
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the Apache License, Version 2.0.
 *
 * Adapted from: https://github.com/dmlc/dlpack
 * DLPack v1.0 legacy tensor layouts.
 */

#pragma once

/**
 * These legacy DLPack declarations let the bindings exchange tensor memory
 * without depending on a producer's tensor library.
 *
 * See https://dmlc.github.io/dlpack/latest/c_api.html
 */

#include <cstdint>

extern "C" {

#define ET_DLPACK_MAJOR 1
#define ET_DLPACK_MINOR 0

/// CPU and CUDA device types from the legacy DLPack ABI.
typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLCUDAManaged = 13,
} DLDeviceType;

typedef struct {
  /// A DLDeviceType value stored as a 32-bit integer.
  int32_t device_type;
  /// Which device of that kind, counting from zero.
  int32_t device_id;
} DLDevice;

/// What a single element is.
typedef enum {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLOpaqueHandle = 3,
  kDLBfloat = 4,
  kDLComplex = 5,
  kDLBool = 6,
} DLDataTypeCode;

typedef struct {
  /// One of DLDataTypeCode.
  uint8_t code;
  /// How wide one element is, in bits.
  uint8_t bits;
  /// How many of them are packed together. Anything but one is a vector type,
  /// which the runtime has no tensor for.
  uint16_t lanes;
} DLDataType;

typedef struct {
  /// The start of the memory. Add byte_offset to reach the first element.
  void* data;
  DLDevice device;
  /// How many dimensions shape and strides have.
  int32_t ndim;
  DLDataType dtype;
  const int64_t* shape;
  /// Strides in ELEMENTS, not bytes. Null means the elements are packed in the
  /// order the shape implies.
  const int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

/**
 * A DLTensor plus who cleans it up.
 *
 * The producer fills this in and hands it over inside a capsule named
 * "dltensor". Whoever takes it renames the capsule to "used_dltensor", so the
 * producer knows not to free it, and calls `deleter` when finished. That is the
 * whole ownership contract, and it is why this is a better fit than the buffer
 * protocol for memory the host cannot even read.
 */
typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  /// The producer's own bookkeeping. Untouched by the consumer.
  void* manager_ctx;
  /// Called by the consumer when it is done. May be null if there is nothing to
  /// release.
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

} // extern "C"
