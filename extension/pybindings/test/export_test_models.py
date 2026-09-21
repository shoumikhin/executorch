# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Writes the .pte files that test_run_exported_models.py loads.

Exporting a model needs torch. Running one does not, and proving that is the point
of the torch-free bindings. So the two halves are split across processes: whoever
has torch runs this and leaves the models on disk, and the tests read them back in
an interpreter that has no torch at all.

    python -m executorch.extension.pybindings.test.export_test_models <directory>
    EXECUTORCH_TEST_MODEL_DIR=<directory> pytest extension/pybindings/test

The models are the ones the rest of the suite already uses, so the two halves agree
on what a method is supposed to do.
"""

import argparse
import contextlib
import io
import os
import sys
from typing import Dict

import torch

from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
    ModuleAddDtype,
    ModuleChannelsLast,
    ModuleMulti,
    ModuleScale4d,
)

# Name to the eager module it comes from. The name is what a test asks for, and it
# is also the file name, so a missing file names the model that is missing.
MODELS = {
    "add": ModuleAdd,
    "multi": ModuleMulti,
    "scale4d": ModuleScale4d,
    "channels_last": ModuleChannelsLast,
    "int64": lambda: ModuleAddDtype(torch.int64),
}


def export_models(into: str) -> Dict[str, str]:
    """Writes every model into `into`, and returns where each one landed."""
    os.makedirs(into, exist_ok=True)
    paths = {}
    for name, module in MODELS.items():
        # create_program prints the whole program, which would bury this script's
        # own output and every other line of the job log around it.
        with contextlib.redirect_stdout(io.StringIO()):
            program, _ = create_program(module())
        paths[name] = os.path.join(into, name + ".pte")
        with open(paths[name], "wb") as file:
            file.write(program.buffer)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="where to write the .pte files")
    arguments = parser.parse_args(argv)
    for name, path in sorted(export_models(arguments.directory).items()):
        print(f"{name}: {path} ({os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
