# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


macro(load_platform_configurations PLATFORM_TUPLE)
  if(NOT EXECUTORCH_PLATFORM_TARGET_TUPLE_CONFIGURATION)
    set(
      EXECUTORCH_PLATFORM_TARGET_TUPLE_CONFIGURATION
      "${PROJECT_SOURCE_DIR}/tools/cmake/platform/${PLATFORM_TUPLE}.cmake"
      CACHE STRING
      "Platform dependent configuration file"
    )
  endif()

  if(NOT EXISTS ${EXECUTORCH_PLATFORM_TARGET_TUPLE_CONFIGURATION})
    message(FATAL_ERROR "Platform configuration file ${EXECUTORCH_PLATFORM_TARGET_TUPLE_CONFIGURATION} not found")
  endif()

  announce_configured_options(EXECUTORCH_PLATFORM_TARGET_TUPLE_CONFIGURATION)
  include(${EXECUTORCH_PLATFORM_TARGET_TUPLE_CONFIGURATION})
endmacro()

# Do not define a config outside of this file.
macro(_define_executorch_config NAME DEFAULT_VALUE DESCRIPTION)
  set(${NAME} ${DEFAULT_VALUE} CACHE STRING ${DESCRIPTION})
  announce_configured_options(${NAME})
endmacro()

macro(set_executorch_config NAME VALUE)
  if(NOT ${NAME})
    set(${NAME} ${VALUE})
  endif()
endmacro()


_define_executorch_config(EXECUTORCH_BUILD_COREML OFF "Enable CoreML pybindings")
