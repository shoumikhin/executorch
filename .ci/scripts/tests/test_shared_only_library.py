# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for shared-only libraries in the Buck macro layer.

A library whose static initializer must run once per process, such as the CUDA
allocator that registers itself, must never be linked as a second static copy. The
open source Buck build cannot evaluate the CUDA targets, so the real macro text is
executed directly, the same way test_is_aten_target.py does.
"""

import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD = REPO_ROOT / "shim_et" / "xplat" / "executorch" / "build"
CUDA_TARGETS = REPO_ROOT / "extension" / "cuda" / "targets.bzl"


def _load_function(path: Path, name: str, next_name: str, namespace: dict):
    text = path.read_text()
    start = text.index(f"def {name}(")
    end = text.index(f"def {next_name}(", start)
    # Padding keeps tracebacks and coverage on the real line numbers.
    source = "\n" * text.count("\n", 0, start) + text[start:end]
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[name]


def _load_cxx_library(calls: list):
    """The real wrapper, with the open source force_static default applied."""
    patch_force_static = _load_function(
        BUILD / "env_interface.bzl",
        "_patch_force_static",
        "_remove_platform_specific_args",
        {},
    )

    def record(*args, **kwargs):
        calls.append(patch_force_static(dict(kwargs)))

    return _load_function(
        BUILD / "runtime_wrapper.bzl",
        "_cxx_library",
        "_cxx_binary_helper",
        {"_cxx_library_common": record},
    )


class TestSharedOnlyLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        self.cxx_library = _load_cxx_library(self.calls)

    def test_static_name_forwards_to_the_shared_library(self) -> None:
        self.cxx_library(
            name="lib",
            srcs=["lib.cpp"],
            preferred_linkage="shared",
            platforms=["CXX"],
            visibility=["//foo/..."],
        )

        library, static = self.calls
        self.assertEqual("lib", library["name"])
        self.assertEqual("shared", library["preferred_linkage"])
        self.assertNotIn("force_static", library)
        self.assertEqual(
            {
                "name": "lib_static",
                "exported_deps": [":lib"],
                "platforms": ["CXX"],
                "visibility": ["//foo/..."],
                "force_static": True,
            },
            static,
        )

    def test_without_static_target(self) -> None:
        self.cxx_library(
            name="lib",
            srcs=["lib.cpp"],
            preferred_linkage="shared",
            define_static_target=False,
        )

        self.assertEqual(["lib"], [call["name"] for call in self.calls])

    def test_other_linkage_still_gets_a_static_copy(self) -> None:
        for linkage in [None, "static"]:
            with self.subTest(linkage=linkage):
                self.calls.clear()
                kwargs = {"name": "lib", "srcs": ["lib.cpp"]}
                if linkage:
                    kwargs["preferred_linkage"] = linkage

                self.cxx_library(**kwargs)

                library, static = self.calls
                self.assertEqual("any", library["preferred_linkage"])
                self.assertEqual(["lib.cpp"], static["srcs"])
                self.assertEqual("static", static["preferred_linkage"])
                self.assertTrue(static["use_static_deps"])


class TestCudaAllocatorTarget(unittest.TestCase):
    def test_registration_is_compiled_into_one_shared_library(self) -> None:
        calls = []
        text = CUDA_TARGETS.read_text()
        source = "\n".join(
            "" if line.startswith("load(") else line for line in text.splitlines()
        )
        namespace = {
            "runtime": types.SimpleNamespace(cxx_library=_load_cxx_library(calls))
        }
        exec(compile(source, str(CUDA_TARGETS), "exec"), namespace)
        namespace["define_common_targets"]()

        compiled = [c for c in calls if "cuda_allocator.cpp" in c.get("srcs", [])]
        self.assertEqual(["cuda_allocator"], [c["name"] for c in compiled])
        self.assertEqual("shared", compiled[0]["preferred_linkage"])
        self.assertFalse(compiled[0].get("force_static", False))

        (static,) = [c for c in calls if c["name"] == "cuda_allocator_static"]
        self.assertEqual([":cuda_allocator"], static["exported_deps"])


if __name__ == "__main__":
    unittest.main()
