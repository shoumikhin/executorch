# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""End to end proof that the bindings run a model with no torch in the process.

Why this file is shaped the way it is. A test that imports torch cannot observe
torch-free behaviour, because the thing it is meant to prove is what happens when
torch is absent, and importing it changes the answer. So the work is split:

  this process     exports the models, because exporting needs torch
  a child process  imports the bindings with torch blocked, and asserts

The child blocks torch with an import hook rather than a second environment, so the
suite proves the same thing whether or not the machine happens to have a torch-free
interpreter lying around. It also checks that no torch library is mapped into the
child, which is the part an import hook cannot fake: a build that linked torch would
pull those libraries in through the extension no matter what Python was told.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

try:
    from executorch.extension.pybindings import portable_lib as runtime
except ImportError:
    from executorch.extension.pybindings import aten_lib as runtime  # @manual

# Exporting needs torch. Asserting does not, and does it in a child.
try:
    import torch

    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.passes import MemoryPlanningPass

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# The child runs this first. It makes `import torch` fail the way it would fail on a
# machine that never installed torch, and leaves nothing in sys.modules to find.
BLOCK_TORCH = """
import sys


class NoTorch:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.partition(".")[0]
        if root in ("torch", "torchvision", "torchaudio"):
            raise ImportError("blocked: this test proves the bindings run without " + root)
        return None


sys.meta_path.insert(0, NoTorch())
"""

# How the child reports which shared libraries the process has actually mapped. An
# import hook cannot hide these, so this is what catches a build that links torch.
# A platform with no way to ask has to raise: answering "nothing is mapped" would
# turn the one assertion a hook cannot fake into one that always holds.
MAPPED_LIBRARIES = """
def mapped_torch_libraries():
    names = set()
    if sys.platform == "linux":
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "/" in line:
                    names.add(line.rsplit("/", 1)[-1].strip())
    elif sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        for index in range(libc._dyld_image_count()):
            names.add(libc._dyld_get_image_name(index).decode().rsplit("/", 1)[-1])
    else:
        raise AssertionError(f"No loaded-library check for {sys.platform}")
    return sorted(
        n for n in names
        if re.match(r"^(?:lib)?(?:torch|c10)(?:[_.-]|$)", n.lower())
    )
"""


def run_without_torch(body, models):
    """Runs `body` in a child that cannot import torch, and returns what it printed.

    The child prints one JSON object on its last line. Anything it raises comes back
    as a failure here with the child's own traceback, because a silent child is worse
    than a noisy one.
    """
    script = "\n".join(
        [
            BLOCK_TORCH,
            "import json, re, sys",
            MAPPED_LIBRARIES,
            "import numpy",
            "from executorch.extension.pybindings import portable_lib as runtime",
            f"MODELS = {models!r}",
            "result = {}",
            textwrap.dedent(body),
            # numpy scalars are not JSON, and asking every case to convert its own
            # would put noise in every assertion below.
            "print(json.dumps(result, default=lambda v: v.item() "
            "if hasattr(v, 'item') else repr(v)))",
        ]
    )
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        # A source checkout on the path would shadow the installed extension.
        cwd=tempfile.gettempdir(),
    )
    if finished.returncode != 0:
        raise AssertionError(
            "the child failed with code {}\n{}".format(
                finished.returncode, finished.stdout + finished.stderr
            )
        )
    return json.loads(finished.stdout.strip().splitlines()[-1])


class TorchFreeEndToEnd(unittest.TestCase):
    """Proves a model runs, correctly, with no torch anywhere in the process."""

    @classmethod
    def setUpClass(cls):
        if not HAS_TORCH:
            raise unittest.SkipTest("the models are exported here, which needs torch")
        cls.directory = tempfile.TemporaryDirectory()
        cls.models = cls._export_models(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "directory"):
            cls.directory.cleanup()

    @staticmethod
    def _export_models(into):
        """Writes the models the child needs, and returns where each one landed."""

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Double(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        class KeepsChannelsLast(torch.nn.Module):
            # A plain multiply is normalised to the default layout during export, so
            # this uses the operator the rest of the suite uses to get a method that
            # really declares channels-last.
            def forward(self, x):
                return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

        unplanned = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            )
        )

        def save(name, module, example, config=None):
            program = to_edge(torch.export.export(module, example))
            buffer = program.to_executorch(
                **({"config": config} if config else {})
            ).buffer
            path = os.path.join(into, name + ".pte")
            with open(path, "wb") as file:
                file.write(buffer)
            return path

        two = (torch.ones(2, 2), torch.full((2, 2), 2.0))
        channels_last = (
            torch.arange(24, dtype=torch.float32)
            .reshape(1, 2, 3, 4)
            .to(memory_format=torch.channels_last),
        )
        return {
            "add": save("add", Add(), two),
            "add_unplanned": save("add_unplanned", Add(), two, unplanned),
            "double": save("double", Double(), (torch.ones(2, 2),)),
            "identity": save("identity", Identity(), (torch.ones(2, 2),), unplanned),
            "channels_last": save("channels_last", KeepsChannelsLast(), channels_last),
            "bfloat16": save(
                "bfloat16", Add(), (torch.ones(2, 3, dtype=torch.bfloat16),) * 2
            ),
            "int64": save("int64", Add(), (torch.ones(2, 2, dtype=torch.int64),) * 2),
            "empty": save("empty", Double(), (torch.ones(2, 0),)),
        }

    def test_the_build_under_test_links_no_torch(self):
        # Everything below is only meaningful for the build this feature adds. If a
        # job asked for that build and got the other one, that is a failure and not
        # a reason to skip, which is how a miswired build hides.
        self.require_torch_free()
        self.assertFalse(runtime._links_torch)

    def test_no_torch_is_imported_or_mapped_while_a_model_runs(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            result["before"] = mapped_torch_libraries()
            module = runtime._load_for_executorch(MODELS["add"])
            ones = numpy.ones((2, 2), numpy.float32)
            twos = numpy.full((2, 2), 2.0, numpy.float32)
            out = module([ones, twos])[0]
            result["values"] = numpy.asarray(out).ravel().tolist()
            result["after"] = mapped_torch_libraries()
            result["torch_in_modules"] = "torch" in sys.modules
            result["links_torch"] = runtime._C._links_torch
            """,
            self.models,
        )
        self.assertEqual(result["values"], [3.0, 3.0, 3.0, 3.0])
        self.assertEqual(result["before"], [])
        self.assertEqual(result["after"], [], "a torch library was mapped in")
        self.assertFalse(result["torch_in_modules"])
        self.assertFalse(result["links_torch"])

    def test_every_way_in_agrees_with_the_model(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            module = runtime._load_for_executorch(MODELS["double"])
            source = numpy.arange(4, dtype=numpy.float32).reshape(2, 2)
            expected = (source * 2).ravel().tolist()

            class OnlyDlpack:
                def __init__(self, array):
                    self._array = array

                def __dlpack__(self, stream=None):
                    return self._array.__dlpack__(stream=stream)

                def __dlpack_device__(self):
                    return self._array.__dlpack_device__()

            ways = {
                "numpy": source,
                "memoryview": memoryview(source),
                "dlpack": OnlyDlpack(source),
                "bare": source,
            }
            result["expected"] = expected
            result["ways"] = {
                name: numpy.asarray(
                    module([value])[0] if name != "bare" else module(value)[0]
                ).ravel().tolist()
                for name, value in ways.items()
            }
            """,
            self.models,
        )
        for name, values in result["ways"].items():
            self.assertEqual(values, result["expected"], f"{name} disagrees")

    def test_dtypes_including_one_no_format_code_can_name(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            integers = runtime._load_for_executorch(MODELS["int64"])
            ones = numpy.ones((2, 2), numpy.int64)
            result["int64"] = numpy.asarray(integers([ones, ones])[0]).ravel().tolist()

            # bfloat16 has no buffer format code, so it travels as raw bytes and the
            # method supplies the dtype and the shape.
            halves = runtime._load_for_executorch(MODELS["bfloat16"])
            raw = numpy.frombuffer(bytes([0, 63]) * 6, dtype=numpy.uint8)
            out = halves([raw, raw])[0]
            result["bfloat16_shape"] = list(out.shape)
            result["bfloat16_dtype"] = out.dtype
            result["bfloat16_declared_dtype"] = (
                halves.method_meta("forward").output_tensor_meta(0).dtype()
            )
            viewed = numpy.asarray(out)
            result["bfloat16_element_bytes"] = viewed.dtype.itemsize
            result["bfloat16_kept_its_shape"] = list(viewed.shape) == [2, 3]
            # numpy has no bfloat16, so the numbers are read as the bytes they are.
            result["bfloat16_bytes"] = list(viewed.tobytes())

            empty = runtime._load_for_executorch(MODELS["empty"])
            result["empty_shape"] = list(empty([numpy.ones((2, 0), numpy.float32)])[0].shape)
            """,
            self.models,
        )
        self.assertEqual(result["int64"], [2, 2, 2, 2])
        self.assertEqual(result["bfloat16_shape"], [2, 3])
        self.assertEqual(result["bfloat16_element_bytes"], 2)
        self.assertTrue(result["bfloat16_kept_its_shape"])
        # The dtype the method declares is the only thing that can name bfloat16 here,
        # so a result that reported any other one would be a result nobody can read.
        self.assertEqual(result["bfloat16_dtype"], result["bfloat16_declared_dtype"])
        # 0x3f00 is 0.5 and 0x3f80 is 1.0, so the model really added the two inputs.
        self.assertEqual(result["bfloat16_bytes"], [0x80, 0x3F] * 6)
        self.assertEqual(result["empty_shape"], [2, 0])

    def test_a_channels_last_method_takes_a_channels_last_input(self):
        self.require_torch_free()
        # What the model itself computes, in logical order, which is what a reader
        # gets back whatever the layout underneath is.
        source = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        expected = (
            torch.nn.functional.interpolate(
                source.to(memory_format=torch.channels_last),
                scale_factor=2,
                mode="nearest",
            )
            .flatten()
            .tolist()
        )
        result = run_without_torch(
            """
            module = runtime._load_for_executorch(MODELS["channels_last"])
            source = numpy.arange(24, dtype=numpy.float32).reshape(1, 2, 3, 4)
            channels_last = numpy.ascontiguousarray(source.transpose(0, 2, 3, 1)).transpose(
                0, 3, 1, 2
            )
            out = module([channels_last])[0]
            result["values"] = numpy.asarray(out).reshape(-1).tolist()
            contiguous = numpy.ascontiguousarray(source)
            try:
                module([contiguous])
                result["contiguous"] = "accepted"
            except RuntimeError as error:
                result["contiguous"] = "refused"
            """,
            self.models,
        )
        self.assertEqual(result["values"], expected)
        # The method was exported channels-last, so the other layout holds the same
        # numbers in a different order and has to be refused rather than misread.
        self.assertEqual(result["contiguous"], "refused")

    def test_the_caller_keeps_owning_its_memory(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            method = runtime._load_program(MODELS["add_unplanned"]).load_method("forward")
            ones = numpy.ones((2, 2), numpy.float32)
            twos = numpy.full((2, 2), 2.0, numpy.float32)

            # Nothing is copied, so a change made before running is one the method sees.
            method.set_inputs([ones, twos])
            ones[:] = 5.0
            method.execute()
            result["read_in_place"] = numpy.asarray(method.get_outputs()[0]).ravel()[0]

            # Memory that moves after it was handed over must be refused, not read.
            moving = numpy.ones((2, 2), numpy.float32)
            method.set_inputs([moving, twos])
            moving.resize(4096, refcheck=False)
            try:
                method.execute()
                result["moved"] = "ran"
            except RuntimeError:
                result["moved"] = "refused"

            # And nothing may be handed back from a call that was refused.
            try:
                method.get_outputs()
                result["outputs_after_refusal"] = "returned"
            except RuntimeError:
                result["outputs_after_refusal"] = "refused"

            # Read only memory is never lent to something that may write to it.
            frozen = bytes(numpy.ones((2, 2), numpy.float32).tobytes())
            view = numpy.frombuffer(frozen, dtype=numpy.float32).reshape(2, 2)
            method.set_inputs([view, twos])
            method.execute()
            result["read_only_intact"] = frozen == bytes(
                numpy.ones((2, 2), numpy.float32).tobytes()
            )
            """,
            self.models,
        )
        self.assertEqual(result["read_in_place"], 7.0)
        self.assertEqual(result["moved"], "refused")
        self.assertEqual(result["outputs_after_refusal"], "refused")
        self.assertTrue(result["read_only_intact"])

    def test_results_are_independent_and_readable_twice(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            method = runtime._load_program(MODELS["add_unplanned"]).load_method("forward")
            twos = numpy.full((2, 2), 2.0, numpy.float32)
            first = method([numpy.ones((2, 2), numpy.float32), twos])[0]
            kept = numpy.asarray(first).copy().ravel().tolist()
            method([numpy.full((2, 2), 9.0, numpy.float32), twos])
            result["earlier_result_unchanged"] = numpy.asarray(first).ravel().tolist() == kept

            method.set_inputs([numpy.ones((2, 2), numpy.float32), twos])
            method.execute()
            result["same_object_twice"] = (
                method.get_outputs()[0] is method.get_outputs()[0]
            )

            # A result handed straight back in has to work, and a method that returns
            # its input must return the input's values rather than an empty buffer.
            identity = runtime._load_program(MODELS["identity"]).load_method("forward")
            source = numpy.full((2, 2), 4.0, numpy.float32)
            result["identity"] = numpy.asarray(identity([source])[0]).ravel().tolist()
            result["fed_back"] = numpy.asarray(
                identity([identity([source])[0]])[0]
            ).ravel().tolist()
            """,
            self.models,
        )
        self.assertTrue(result["earlier_result_unchanged"])
        self.assertTrue(result["same_object_twice"])
        self.assertEqual(result["identity"], [4.0, 4.0, 4.0, 4.0])
        self.assertEqual(result["fed_back"], [4.0, 4.0, 4.0, 4.0])

    def test_mistakes_are_refused_and_say_why(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            module = runtime._load_for_executorch(MODELS["add"])
            ones = numpy.ones((2, 2), numpy.float32)
            cases = {
                "wrong dtype": [ones.astype(numpy.float64), ones],
                "too few inputs": [ones],
                "a size the runtime cannot hold": [
                    numpy.empty((2**31 + 5, 0), numpy.float32), ones
                ],
                "a rank past the limit": [numpy.ones((1,) * 20, numpy.float32), ones],
                "a strided view": [numpy.ones((4, 4), numpy.float32)[::2, ::2], ones],
                "not memory at all": ["a string", ones],
            }
            result["refusals"] = {}
            for name, inputs in cases.items():
                try:
                    module(inputs)
                    result["refusals"][name] = None
                except Exception as error:
                    result["refusals"][name] = [type(error).__name__, str(error).strip()]
            """,
            self.models,
        )
        # The text each refusal has to carry. A wrong error kind, or a message that
        # stops naming the problem, then fails here instead of passing quietly.
        wanted = {
            "wrong dtype": "the method expects Float",
            # This one comes from the runtime's own execute, which reports only that
            # it failed, so there is no better text to pin here yet.
            "too few inputs": "Failed to execute method forward",
            "a size the runtime cannot hold": "outside what the runtime can describe",
            "a rank past the limit": "strides no runtime layout describes",
            "a strided view": "strides no runtime layout describes",
            "not memory at all": "inputs are passed as a flat list of tensors",
        }
        for name, fragment in wanted.items():
            refusal = result["refusals"][name]
            self.assertIsNotNone(refusal, f"{name} was accepted")
            kind, message = refusal
            self.assertEqual(kind, "RuntimeError", f"{name} raised {kind}: {message}")
            self.assertIn(fragment, message, f"{name} said: {message}")

    def test_a_program_read_from_bytes_nobody_else_holds(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            with open(MODELS["add"], "rb") as file:
                # The temporary is the point: nothing but the loader holds these bytes.
                method = runtime._load_program_from_buffer(file.read()).load_method("forward")
            ones = numpy.ones((2, 2), numpy.float32)
            twos = numpy.full((2, 2), 2.0, numpy.float32)
            result["values"] = numpy.asarray(method([ones, twos])[0]).ravel().tolist()
            """,
            self.models,
        )
        self.assertEqual(result["values"], [3.0, 3.0, 3.0, 3.0])

    def test_many_programs_and_methods_in_one_process(self):
        self.require_torch_free()
        result = run_without_torch(
            """
            import gc

            ones = numpy.ones((2, 2), numpy.float32)
            twos = numpy.full((2, 2), 2.0, numpy.float32)
            values = []
            for _ in range(50):
                module = runtime._load_for_executorch(MODELS["add"])
                values.append(numpy.asarray(module([ones, twos])[0]).ravel()[0])
                del module
            gc.collect()
            result["all_the_same"] = len(set(float(v) for v in values)) == 1
            result["value"] = float(values[0])

            # Two methods of two programs, alive at once, each still its own.
            adder = runtime._load_program(MODELS["add"]).load_method("forward")
            doubler = runtime._load_program(MODELS["double"]).load_method("forward")
            result["interleaved"] = [
                float(numpy.asarray(adder([ones, twos])[0]).ravel()[0]),
                float(numpy.asarray(doubler([ones])[0]).ravel()[0]),
                float(numpy.asarray(adder([ones, twos])[0]).ravel()[0]),
            ]
            """,
            self.models,
        )
        self.assertTrue(result["all_the_same"])
        self.assertEqual(result["value"], 3.0)
        self.assertEqual(result["interleaved"], [3.0, 2.0, 3.0])

    def require_torch_free(self):
        if os.environ.get("EXECUTORCH_TEST_WITHOUT_TORCH") == "1":
            self.assertFalse(
                runtime._links_torch,
                "CI requested a torch-free build, but the extension links torch",
            )
        elif runtime._links_torch:
            self.skipTest("this build links torch")


if __name__ == "__main__":
    unittest.main()
