# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Backends
_define_executorch_config(EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" OFF)
_define_executorch_config(EXECUTORCH_BUILD_COREML "Enable CoreML backend" OFF)
_define_executorch_config(EXECUTORCH_BUILD_MPS "Build the MPS backend" OFF)

# Targets
_define_executorch_config(EXECUTORCH_BUILD_PYBIND "Build the Python binding" OFF)
_define_executorch_config(EXECUTORCH_BUILD_EXECUTOR_RUNNER "Build the executor_runner executable" ON)

if(EXECUTORCH_BUILD_PYBIND)
  set(_default_executorch_build_extension_tensor ON)
else()
  set(_default_executorch_build_extension_tensor OFF)
endif()
_define_executorch_config(
  EXECUTORCH_BUILD_EXTENSION_TENSOR
  "Build the Tensor extension"
  ${_default_executorch_build_extension_tensor}
)
