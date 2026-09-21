# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""What the wrapper says when the extension will not load.

portable_lib imports torch when the extension fails to import, because a wheel's
extension that links libtorch cannot resolve it until torch has loaded it. Any other
reason the extension might fail takes the same branch, so the error that says what
actually went wrong has to survive that second attempt instead of being replaced by
a message about torch.
"""

import subprocess
import sys
import tempfile
import textwrap
import unittest

# What a broken build looks like: the extension is there and cannot be loaded, and
# torch has nothing to do with it.
BROKEN_EXTENSION = "dlopen failed: symbol not found in flat namespace (_not_a_symbol)"

CHILD = textwrap.dedent(
    f"""\
    import sys


    class Broken:
        def find_spec(self, name, path=None, target=None):
            if name == "executorch.extension.pybindings._C":
                raise ImportError({BROKEN_EXTENSION!r}, name=name)
            # Also stand in for a machine that never installed torch, which is where
            # the extension's own error used to disappear.
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return None


    sys.meta_path.insert(0, Broken())
    from executorch.extension.pybindings import portable_lib  # noqa: F401
    """
)


class PortableLibImportTest(unittest.TestCase):
    def test_a_broken_extension_is_not_reported_as_a_missing_torch(self):
        child = subprocess.run(
            [sys.executable, "-c", CHILD],
            capture_output=True,
            text=True,
            # A source checkout on the path would shadow the installed extension.
            cwd=tempfile.gettempdir(),
        )

        self.assertNotEqual(
            child.returncode, 0, "importing a broken extension should have failed"
        )
        self.assertIn(
            BROKEN_EXTENSION,
            child.stderr,
            "the reason the extension would not load is missing from the failure",
        )
        self.assertIn(
            "No module named 'torch'",
            child.stderr,
            "the fallback that loads libtorch through torch was not attempted",
        )


if __name__ == "__main__":
    unittest.main()
