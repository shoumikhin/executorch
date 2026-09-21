# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Runs models that were exported somewhere else, with numpy and nothing else.

Every other test in this directory exports the model it runs, so every one of them
needs torch, so none of them can run in the interpreter this feature exists for. The
models here come from disk instead: export_test_models.py writes a directory of
<name>.pte files with torch, and these tests load them without it.

Point EXECUTORCH_TEST_MODEL_DIR at that directory. Without it, an interpreter that
has torch exports the models itself, so running the suite by hand needs no extra
step. An interpreter with no torch cannot, and it fails and says what to run, since
a leg that skips its way to green is worse than no leg at all.
"""

import os
import tempfile
import unittest

import numpy

from executorch.runtime import Runtime

try:
    from executorch.extension.pybindings import portable_lib as bindings
except ImportError:
    from executorch.extension.pybindings import aten_lib as bindings  # @manual

MODEL_DIR_VARIABLE = "EXECUTORCH_TEST_MODEL_DIR"


def in_channels_last(values):
    """The same values, held in memory the way a channels-last method wants them."""
    return numpy.ascontiguousarray(values.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


class RunExportedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = None
        directory = os.environ.get(MODEL_DIR_VARIABLE)
        if directory is None:
            try:
                from executorch.extension.pybindings.test.export_test_models import (
                    export_models,
                )
            except ImportError as error:
                raise AssertionError(
                    f"there are no models to run: set {MODEL_DIR_VARIABLE} to a "
                    "directory written by "
                    "`python -m executorch.extension.pybindings.test."
                    f"export_test_models <directory>`, because this interpreter "
                    f"cannot export them itself ({error})"
                ) from error
            cls.temporary = tempfile.TemporaryDirectory()
            directory = cls.temporary.name
            export_models(directory)
        cls.directory = directory

    @classmethod
    def tearDownClass(cls):
        if cls.temporary is not None:
            cls.temporary.cleanup()

    def model(self, name):
        """The path to one exported model, or a failure naming the one that is gone."""
        path = os.path.join(self.directory, name + ".pte")
        self.assertTrue(
            os.path.isfile(path),
            f"{path} was never exported: run export_test_models against "
            f"{self.directory}",
        )
        return path

    def test_the_documented_api_runs_a_model_and_returns_its_numbers(self):
        method = Runtime.get().load_program(self.model("add")).load_method("forward")
        ones = numpy.ones((2, 2), numpy.float32)
        twos = numpy.full((2, 2), 2.0, numpy.float32)

        result = method.execute([ones, twos])[0]

        values = numpy.asarray(result)
        self.assertEqual(values.tolist(), [[3.0, 3.0], [3.0, 3.0]])
        self.assertEqual(values.dtype, numpy.float32)
        # The result is the runtime's own memory, so reading it again reads the same
        # bytes rather than a copy of them.
        self.assertEqual(
            numpy.asarray(result).__array_interface__["data"][0],
            values.__array_interface__["data"][0],
        )

    def test_the_module_loader_agrees_with_the_program_loader(self):
        module = bindings._load_for_executorch(self.model("add"))
        ones = numpy.ones((2, 2), numpy.float32)
        twos = numpy.full((2, 2), 2.0, numpy.float32)

        result = numpy.asarray(module([ones, twos])[0])

        self.assertEqual(result.tolist(), [[3.0, 3.0], [3.0, 3.0]])

    def test_a_program_runs_each_of_its_methods(self):
        module = bindings._load_for_executorch(self.model("multi"))
        self.assertEqual(sorted(module.method_names()), ["forward", "forward2"])
        ones = numpy.ones((2, 2), numpy.float32)

        forward = numpy.asarray(module.forward([ones, ones])[0])
        forward2 = numpy.asarray(module.run_method("forward2", [ones, ones])[0])

        self.assertEqual(forward.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertEqual(forward2.tolist(), [[3.0, 3.0], [3.0, 3.0]])

    def test_metadata_describes_the_model(self):
        metadata = (
            Runtime.get()
            .load_program(self.model("add"))
            .load_method("forward")
            .metadata
        )

        self.assertEqual(metadata.name(), "forward")
        self.assertEqual(metadata.num_inputs(), 2)
        self.assertEqual(metadata.num_outputs(), 1)
        for index in range(metadata.num_inputs()):
            self.assertEqual(metadata.input_tensor_meta(index).sizes(), (2, 2))
            self.assertEqual(metadata.input_tensor_meta(index).nbytes(), 16)
        self.assertEqual(metadata.output_tensor_meta(0).sizes(), (2, 2))

    def test_an_integer_model_keeps_its_dtype(self):
        module = bindings._load_for_executorch(self.model("int64"))
        sevens = numpy.full((2, 3), 7, numpy.int64)

        result = numpy.asarray(module([sevens, sevens])[0])

        self.assertEqual(result.dtype, numpy.int64)
        self.assertEqual(result.tolist(), [[14, 14, 14], [14, 14, 14]])

    def test_a_channels_last_method_takes_channels_last_memory(self):
        values = numpy.arange(24, dtype=numpy.float32).reshape(1, 2, 3, 4)
        module = bindings._load_for_executorch(self.model("channels_last"))

        result = numpy.asarray(module(in_channels_last(values))[0])

        # The method doubles each side by repeating a value into a 2x2 block.
        expected = numpy.repeat(numpy.repeat(values, 2, axis=2), 2, axis=3)
        self.assertEqual(result.tolist(), expected.tolist())

    def test_a_default_layout_method_refuses_channels_last_memory(self):
        values = numpy.arange(24, dtype=numpy.float32).reshape(1, 2, 3, 4)
        module = bindings._load_for_executorch(self.model("scale4d"))

        # The same values in the layout the method was exported with are fine.
        self.assertEqual(
            numpy.asarray(module(values)[0]).ravel()[:4].tolist(),
            [0.0, 2.0, 4.0, 6.0],
        )
        # In the other layout they would be taken apart in the wrong order, which
        # used to return wrong numbers with no error.
        self.assertRaises(RuntimeError, module, in_channels_last(values))


if __name__ == "__main__":
    unittest.main()
