# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""API for loading and executing ExecuTorch PTE files using the C++ runtime.

.. warning::

    This API is experimental and subject to change without notice.
"""

import logging
import os
import sys
import warnings as _warnings

# Importing exir requires torch. Reuse its warning category only if already loaded;
# otherwise use its base class. Filters for ExperimentalWarning alone will not match.
_category = DeprecationWarning
if "executorch.exir._warnings" in sys.modules:
    _category = sys.modules["executorch.exir._warnings"].ExperimentalWarning
_warnings.warn(
    "This API is experimental and subject to change without notice.", _category
)

# When installed as a pip wheel, the extension needs libtorch loaded before it
# can resolve its own dependencies, and importing torch is what loads it. An
# extension built without the torch tensor path links no libtorch and must not
# pay for that import, so the extension is tried first and torch is only brought
# in if it turns out to be needed.
_extension_error = None
try:
    from executorch.extension.pybindings import _C

    # An extension built before this flag existed does not report it. Assume the
    # old answer, which is that it needs torch, rather than failing the import.
    _needs_torch = getattr(_C, "_links_torch", True)
except ImportError as _error:
    # libtorch is what the extension could not resolve, and importing torch loads
    # it. Any other cause lands here too, and this is the only error that says what
    # it was, so keep it rather than let a message about torch replace it.
    _extension_error = _error
    _needs_torch = True

if _needs_torch:
    try:
        # Torch's tensor converters need its Python types registered before use.
        import torch as _torch  # noqa: F401

        from executorch.extension.pybindings import _C
    except ImportError as _error:
        raise _error from _extension_error

_links_torch = getattr(_C, "_links_torch", True)

logger = logging.getLogger(__name__)

# Auto-discover the OpenVINO C library path from the pip-installed openvino
# package so the C++ backend's dlopen("libopenvino_c.so") works without the
# user having to set LD_LIBRARY_PATH or OPENVINO_LIB_PATH manually.
if not os.environ.get("OPENVINO_LIB_PATH"):
    try:
        import glob
        import importlib.util

        spec = importlib.util.find_spec("openvino")
        if spec is not None and spec.submodule_search_locations:
            _ov_dir = spec.submodule_search_locations[0]
            _ov_libs = sorted(
                glob.glob(os.path.join(_ov_dir, "libs", "libopenvino_c.so*"))
            )
            if _ov_libs:
                os.environ["OPENVINO_LIB_PATH"] = _ov_libs[0]
            else:
                logger.warning(
                    "OpenVINO package found but libopenvino_c.so not in %s; "
                    "set OPENVINO_LIB_PATH manually if needed",
                    os.path.join(_ov_dir, "libs"),
                )
            del _ov_libs, _ov_dir, spec
    except Exception as e:
        logger.debug("OpenVINO auto-discovery failed: %s", e)

# Update the DLL search path on Windows. This is the recommended way to handle native
# extensions.
if sys.platform == "win32":
    try:
        # The extension DLL should be in the same directory as this file.
        pybindings_dir = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(pybindings_dir)
    except Exception as e:
        logger.error(
            "Failed to add the pybinding extension DLL to the search path. "
            "The extension may not work: %s",
            e,
        )

# Let users import everything from the C++ _C extension as if this
# python file defined them. Although we could import these dynamically, it
# wouldn't preserve the static type annotations.
#
# Note that all of these are experimental, and subject to change without notice.
from executorch.extension.pybindings._C import (  # noqa: F401
    # Disable "imported but unused" (F401) checks.
    _create_profile_block,  # noqa: F401
    _dump_profile_results,  # noqa: F401
    _get_operator_names,  # noqa: F401
    _get_registered_backend_names,  # noqa: F401
    _is_available,  # noqa: F401
    _load_bundled_program_from_buffer,  # noqa: F401
    _load_for_executorch,  # noqa: F401
    _load_for_executorch_from_buffer,  # noqa: F401
    _load_for_executorch_from_bundled_program,  # noqa: F401
    _load_for_executorch_from_data_loader,  # noqa: F401
    _load_program,  # noqa: F401
    _load_program_from_buffer,  # noqa: F401
    _reset_profile_results,  # noqa: F401
    _threadpool_get_thread_count,  # noqa: F401
    _unsafe_reset_threadpool,  # noqa: F401
    BundledModule,  # noqa: F401
    ExecuTorchMethod,  # noqa: F401
    ExecuTorchModule,  # noqa: F401
    ExecuTorchProgram,  # noqa: F401
    MethodMeta,  # noqa: F401
    TensorInfo,  # noqa: F401
    Verification,  # noqa: F401
)

# Clean up so that `dir(portable_lib)` is the same as `dir(_C)`
# (apart from some __dunder__ names).
if "_torch" in globals():
    del _torch
del _category, _extension_error, _needs_torch, _warnings
