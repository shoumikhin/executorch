# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import gc
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import weakref
from io import StringIO

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.backend.test.device_util import DeviceAwarePartitioner
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.schema import DeviceType
    from executorch.extension.pybindings.test.make_test import (
        create_program,
        ModuleAdd,
        ModuleAddConstReturn,
        ModuleAddDtype,
        ModuleAddInPlace,
        ModuleAddSingleInput,
        ModuleAddWithAttributes,
        ModuleChannelsLast,
        ModuleChannelsLastInDefaultOut,
        ModuleLinear,
        ModuleMulti,
        ModuleScale4d,
    )
    from torch.export import export


_CHECK_TORCH_FREE = textwrap.dedent(
    """\
    import ctypes
    import os
    import re
    import sys

    if sys.platform == "linux":
        with open("/proc/self/maps") as maps:
            libraries = [
                line.split(maxsplit=5)[-1].rstrip() for line in maps if "/" in line
            ]
    elif sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        libraries = [
            os.fsdecode(dyld._dyld_get_image_name(i))
            for i in range(dyld._dyld_image_count())
        ]
    elif sys.platform == "win32":
        from ctypes import wintypes

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process = wintypes.HANDLE(-1)
        modules = (wintypes.HMODULE * 4096)()
        needed = wintypes.DWORD()
        assert psapi.EnumProcessModules(
            process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ), ctypes.get_last_error()
        assert needed.value <= ctypes.sizeof(modules), "module list was truncated"
        libraries = []
        for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            name = ctypes.create_unicode_buffer(32768)
            assert psapi.GetModuleFileNameExW(
                process, wintypes.HMODULE(module), name, len(name)
            ), ctypes.get_last_error()
            libraries.append(name.value)
    else:
        raise AssertionError(f"No loaded-library check for {sys.platform}")

    linked = [
        path for path in libraries
        if re.match(r"^(?:lib)?(?:torch|c10)(?:[_.-]|$)", os.path.basename(path).lower())
    ]
    assert not linked, f"torch libraries are mapped: {linked}"
    assert not any(
        name == "torch" or name.startswith("torch.") for name in sys.modules
    ), "the bindings imported torch"
    """
)


_HOSTILE_CAPSULE = textwrap.dedent(
    """\
    import ctypes
    import sys

    from executorch.extension.pybindings.portable_lib import _load_for_executorch


    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


    class DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]


    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.c_void_p),
            ("strides", ctypes.c_void_p),
            ("byte_offset", ctypes.c_uint64),
        ]


    class DLManagedTensor(ctypes.Structure):
        _fields_ = [
            ("dl_tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ]


    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    values = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
    shape = (ctypes.c_int64 * 2)(2, 2)
    # Three of these multiply to two to the 66, which is zero in 64 bits, so an
    # element count worked out by multiplying the sizes reads as no elements.
    wrapping = (ctypes.c_int64 * 3)(1 << 22, 1 << 22, 1 << 22)
    # Nothing here is freed by the capsule, so the structures it points at have to
    # outlive it, and there is no deleter to run.
    alive = [values, shape, wrapping]


    class Producer:
        \"\"\"Hands over exactly the rank, shape and data pointer it is given.\"\"\"

        def __init__(self, ndim, shape_address, data_address):
            self._described = (ndim, shape_address, data_address)

        def __dlpack_device__(self):
            return (1, 0)  # kDLCPU, device zero

        def __dlpack__(self, stream=None):
            ndim, shape_address, data_address = self._described
            managed = DLManagedTensor()
            managed.dl_tensor.data = data_address
            managed.dl_tensor.device = DLDevice(1, 0)
            managed.dl_tensor.ndim = ndim
            managed.dl_tensor.dtype = DLDataType(2, 32, 1)  # kDLFloat, 32 bits
            managed.dl_tensor.shape = shape_address
            managed.dl_tensor.strides = None
            managed.dl_tensor.byte_offset = 0
            alive.append(managed)
            return new_capsule(ctypes.addressof(managed), b"dltensor", None)


    honest = Producer(2, ctypes.addressof(shape), ctypes.addressof(values))
    method = _load_for_executorch(sys.argv[1])
    for described in (
        honest,
        Producer(1 << 20, ctypes.addressof(shape), ctypes.addressof(values)),
        Producer(2, None, ctypes.addressof(values)),
        Producer(2, ctypes.addressof(shape), None),
        Producer(3, ctypes.addressof(wrapping), None),
        Producer(-3, ctypes.addressof(shape), ctypes.addressof(values)),
    ):
        try:
            method([described, honest])
        except Exception as error:
            print(type(error).__name__, str(error).splitlines()[0])
        else:
            print("ran")
    """
)


class _OnlyDlpack:
    """A producer that offers nothing but DLPack, so only that path can read it."""

    def __init__(self, array) -> None:
        self._array = array

    def __dlpack__(self, stream=None):
        return self._array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


def _neg_program(example):
    """A method that negates its input, exported with the example's own layout.

    Negation is the simplest operation the portable kernels run in a
    channels-last layout: an operation taking a second tensor, a scalar included,
    needs that tensor to share the layout, and the exporter lays it out in the
    default order.
    """

    class Neg(torch.nn.Module):
        def forward(self, x):
            return torch.neg(x)

    return to_edge(torch.export.export(Neg(), (example,))).to_executorch()


class PybindingsTestBase(unittest.TestCase):
    def setUp(self):
        # Will test both portable and aten
        kernel_mode = None
        try:
            from executorch.extension.pybindings import portable_lib as runtime

            kernel_mode = "portable"
        except Exception:
            print("can't load portable lib")

        if kernel_mode is None:
            try:
                from executorch.extension.pybindings import (  # noqa: F811
                    aten_lib as runtime,
                )

                kernel_mode = "aten"
            except Exception:
                print("can't load aten lib")

        assert kernel_mode is not None
        # Only the portable build converts an incoming tensor, so a test for
        # that conversion has to skip under the aten build.
        self.kernel_mode = kernel_mode
        self.load_fn = runtime._load_for_executorch_from_buffer
        self.load_prog_fn = runtime._load_program_from_buffer
        self.runtime = runtime

    def require_torch_free(self):
        if os.environ.get("EXECUTORCH_TEST_WITHOUT_TORCH") == "1":
            self.assertFalse(
                self.runtime._links_torch,
                "CI requested a torch-free build, but the extension links torch",
            )
        elif self.runtime._links_torch:
            self.skipTest("this build links torch")


class PybindingsImportTest(PybindingsTestBase):
    def test_torch_free_build_does_not_link_torch(self):
        self.require_torch_free()
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from executorch.extension.pybindings import portable_lib\n"
                + _CHECK_TORCH_FREE,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)


class PybindingsTest(PybindingsTestBase):
    def setUp(self):
        if torch is None:
            self.skipTest("torch is required to export test models")
        super().setUp()

    def test_e2e(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module.forward(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_multiple_entry(self):
        program, inputs = create_program(ModuleMulti())
        executorch_module = self.load_fn(program.buffer)

        executorch_output = executorch_module.forward(inputs)[0]
        self.assertTrue(torch.allclose(executorch_output, torch.ones(2, 2) * 2))

        executorch_output2 = executorch_module.run_method("forward2", inputs)[0]
        self.assertTrue(torch.allclose(executorch_output2, torch.ones(2, 2) * 3))

    def test_output_lifespan(self):
        def lower_function_call():
            program, inputs = create_program(ModuleMulti())
            executorch_module = self.load_fn(program.buffer)
            return executorch_module.forward(inputs)

        outputs = lower_function_call()
        self.assertTrue(torch.allclose(outputs[0], torch.ones(2, 2) * 2))

    def test_module_callable(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_module_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs[0])[0]
        expected = inputs[0] + inputs[0]
        self.assertEqual(str(expected), str(executorch_output))

    def test_stderr_redirect(self):
        class RedirectedStderr:
            def __init__(self):
                self._stderr = None
                self._string_io = None

            def __enter__(self):
                self._stderr = sys.stderr
                sys.stderr = self._string_io = StringIO()
                return self

            def __exit__(self, type, value, traceback):
                sys.stderr = self._stderr

            def __str__(self):
                return self._string_io.getvalue()

        with RedirectedStderr() as out:
            try:
                exported_program, inputs = create_program(ModuleAdd())
                executorch_module = self.load_fn(exported_program.buffer)
                inputs = (*inputs, 1)
                executorch_output = executorch_module(inputs)[0]  # noqa
                self.assertFalse(True)  # should be unreachable
            except Exception:
                self.assertTrue(str(out).find("The length of given input array"))

    def test_quantized_ops(self):
        eager_module = ModuleAdd()

        from executorch.exir import EdgeCompileConfig
        from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.backend_config.executorch import (
            get_executorch_backend_config,
        )
        from torch.ao.quantization.quantize_fx import (
            _convert_to_reference_decomposed_fx,
            prepare_fx,
        )

        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        example_inputs = (
            torch.ones(1, 5, dtype=torch.float32),
            torch.ones(1, 5, dtype=torch.float32),
        )
        m = prepare_fx(
            eager_module,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        m = _convert_to_reference_decomposed_fx(m)
        config = EdgeCompileConfig(_check_ir_validity=False)
        m = to_edge(export(m, example_inputs, strict=True), compile_config=config)
        m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])

        exec_prog = m.to_executorch()

        executorch_module = self.load_fn(exec_prog.buffer)
        executorch_output = executorch_module.forward(example_inputs)[0]

        expected = example_inputs[0] + example_inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_constant_output_not_memory_planned(self):
        exported_program, inputs = create_program(
            ModuleAddConstReturn(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            ),
        )

        exported_program.dump_executorch_program(verbose=True)

        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module((torch.ones(2, 2),))

        expected = torch.ones(2, 2) + torch.ones(2, 2)
        self.assertTrue(torch.allclose(expected, executorch_output[0]))
        self.assertEqual(str(torch.ones(2, 2)), str(executorch_output[1]))

    def test_channels_last(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)

        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_unsupported_dim_order(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        inputs = (torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),)

        executorch_module = self.load_fn(exported_program.buffer)
        self.assertRaises(RuntimeError, executorch_module, inputs[0])

    def test_channels_last_in_default_out(self) -> None:
        model = ModuleChannelsLastInDefaultOut()
        exported_program, inputs = create_program(model)

        executorch_module = self.load_fn(exported_program.buffer)
        executorch_output = executorch_module(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_channels_last_input_for_default_layout_method(self) -> None:
        # The method reads its input in the default order, so the same values in
        # channels-last memory would be taken apart in the wrong one. Accepting
        # it used to return wrong numbers with no error.
        exported_program, inputs = create_program(ModuleScale4d())
        executorch_module = self.load_fn(exported_program.buffer)

        channels_last = inputs[0].to(memory_format=torch.channels_last)
        self.assertRaises(RuntimeError, executorch_module, (channels_last,))

    def test_default_layout_input_for_channels_last_method(self) -> None:
        # The mirror of the test above, and the case the refusal exists for. Every
        # way in holds the source's own layout, so every way in refuses the wrong
        # one and names the stride the method wanted. The right layout still runs,
        # and agrees with what the model computes in eager mode.
        values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        channels_last = values.to(memory_format=torch.channels_last)
        executorch_module = self.load_fn(_neg_program(channels_last).buffer)
        expected = torch.neg(values).numpy()

        for path, source in (
            ("torch", channels_last),
            ("buffer", channels_last.numpy()),
            ("dlpack", _OnlyDlpack(channels_last.numpy())),
        ):
            with self.subTest(path=path, layout="channels-last"):
                actual = np.asarray(executorch_module([source])[0])
                self.assertTrue(np.array_equal(expected, actual))

        for path, source in (
            ("torch", values),
            ("buffer", values.numpy()),
            ("dlpack", _OnlyDlpack(values.numpy())),
        ):
            with self.subTest(path=path, layout="default"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "stride 48 bytes at dimension 1, but the method expects 4",
                ):
                    executorch_module([source])

    def test_a_size_one_dimension_may_be_laid_out_either_way(self) -> None:
        # A dimension of size one is never stepped along, so its stride says
        # nothing about where the values are, and sources disagree about what to
        # report for one. numpy gives a new axis a stride of zero and hands that on
        # through DLPack and through torch, while a tensor of the same shape built
        # directly reports the stride the shape implies. The bytes are the same
        # either way, which is what torch is saying by calling the zero-strided one
        # contiguous, so all of them have to be accepted.
        array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[None]
        zero_stride = torch.from_numpy(array)
        values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        self.assertEqual(zero_stride.stride(), (0, 12, 4, 1))
        self.assertEqual(values.stride(), (24, 12, 4, 1))
        self.assertTrue(zero_stride.is_contiguous())
        self.assertTrue(torch.equal(values, zero_stride))

        executorch_module = self.load_fn(_neg_program(values).buffer)
        expected = torch.neg(values).numpy()
        for path, source in (
            ("torch", values),
            ("torch, zero stride", zero_stride),
            ("dlpack, zero stride", _OnlyDlpack(array)),
        ):
            with self.subTest(path=path):
                actual = np.asarray(executorch_module([source])[0])
                self.assertTrue(np.array_equal(expected, actual))

    def test_an_input_with_no_elements_is_accepted_whatever_its_strides(self) -> None:
        # A tensor with no elements has no layout to get wrong, and what its source
        # reports for strides carries no meaning: numpy says every stride is zero
        # for this shape, torch says the ones the shape implies.
        empty = torch.zeros(2, 0, 3)
        array = np.zeros((2, 0, 3), np.float32)
        self.assertEqual(empty.stride(), (3, 3, 1))
        self.assertEqual(array.strides, (0, 0, 0))

        executorch_module = self.load_fn(_neg_program(empty).buffer)
        for path, source in (
            ("torch", empty),
            ("buffer", array),
            ("dlpack", _OnlyDlpack(array)),
        ):
            with self.subTest(path=path):
                actual = np.asarray(executorch_module([source])[0])
                self.assertEqual(actual.shape, (2, 0, 3))

        # A view with no elements keeps the strides of whatever it was cut from, so
        # they are not the ones this shape implies. The bridge to the runtime's own
        # tensor compares the two, and has to let this through.
        sliced = torch.zeros(2, 4, 3)[:, 0:0, :]
        self.assertEqual(sliced.shape, (2, 0, 3))
        self.assertEqual(sliced.stride(), (12, 3, 1))
        self.assertEqual(np.asarray(executorch_module([sliced])[0]).shape, (2, 0, 3))

    def test_default_layout_input_for_a_rank_five_channels_last_method(self) -> None:
        # The exporter records a channels-last order past four dimensions too,
        # [0, 2, 3, 4, 1] here, so the refusal has to hold at any rank and not
        # only at the four dimensional shape channels-last is usually written for.
        values = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
        executorch_module = self.load_fn(
            _neg_program(values.to(memory_format=torch.channels_last_3d)).buffer
        )

        with self.assertRaisesRegex(
            RuntimeError, "stride 96 bytes at dimension 1, but the method expects 4"
        ):
            executorch_module([values])

    def test_channels_last_input_for_a_rank_five_channels_last_method(self) -> None:
        # The runtime reads a channels-last order at five dimensions as well as at
        # four, and the exporter records one at both, so a rank five channels-last
        # method needs a way in and not only a way to be refused.
        values = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
        channels_last = values.to(memory_format=torch.channels_last_3d)
        executorch_module = self.load_fn(_neg_program(channels_last).buffer)
        expected = torch.neg(values).numpy()

        for path, source in (
            ("torch", channels_last),
            ("buffer", channels_last.numpy()),
            ("dlpack", _OnlyDlpack(channels_last.numpy())),
        ):
            with self.subTest(path=path):
                actual = np.asarray(executorch_module([source])[0])
                self.assertTrue(np.array_equal(expected, actual))

    def test_an_input_of_a_rank_the_method_does_not_expect_is_refused(self) -> None:
        # A rank the method does not agree with cannot be the method's layout, so
        # it is refused here with the rank named, rather than left to fail deeper
        # down as a resize the caller never asked for.
        values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        executorch_module = self.load_fn(_neg_program(values).buffer)

        with self.assertRaisesRegex(
            RuntimeError, "has 5 dimensions, but the method expects 4"
        ):
            executorch_module([torch.zeros(1, 2, 2, 3, 4)])

    def test_buffer_input_matches_torch_input(self) -> None:
        # numpy arrays reach the runtime through the buffer protocol, which is
        # the path a caller without torch installed takes.
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        from_torch = np.asarray(executorch_module(inputs)[0])
        from_buffer = np.asarray(executorch_module([i.numpy() for i in inputs])[0])
        self.assertTrue(np.array_equal(from_torch, from_buffer))

    def test_buffer_input_dtypes(self) -> None:
        # A buffer says what its elements are with a format code, and the method
        # only accepts the dtype it was exported with, so a wrong mapping shows
        # up as a rejected input rather than as wrong numbers.
        for dtype in (
            torch.float32,
            torch.float64,
            torch.float16,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            with self.subTest(dtype=dtype):
                exported_program, inputs = create_program(ModuleAddDtype(dtype))
                executorch_module = self.load_fn(exported_program.buffer)

                expected = np.asarray(executorch_module(inputs)[0])
                actual = np.asarray(executorch_module([i.numpy() for i in inputs])[0])
                self.assertTrue(np.array_equal(expected, actual))

    def test_buffer_input_channels_last(self) -> None:
        # Distinct values, and compared against what the model itself computes, so a
        # layout read the wrong way shows up as wrong numbers. Comparing against
        # another call through the same conversion would agree with itself either way.
        model = ModuleChannelsLast()
        exported_program, _ = create_program(model)
        executorch_module = self.load_fn(exported_program.buffer)

        source = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        channels_last = source.to(memory_format=torch.channels_last)
        self.assertFalse(channels_last.numpy().flags.c_contiguous)
        expected = model(channels_last).contiguous().numpy()

        actual = np.asarray(executorch_module([channels_last.numpy()])[0])
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(np.array_equal(expected, actual))

    def test_buffer_input_unsupported_layout(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        transposed = inputs[0].numpy().transpose()
        self.assertRaisesRegex(
            RuntimeError,
            "no runtime layout describes",
            executorch_module,
            [transposed, transposed],
        )

    def test_buffer_input_unsupported_dtype(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        # An extended float has no format code any runtime dtype maps to, so the
        # buffer cannot be read as one.
        wide = inputs[0].numpy().astype(np.longdouble)
        self.assertRaisesRegex(
            RuntimeError,
            "no dtype describes",
            executorch_module,
            [wide, wide],
        )

        # A dtype the method does not expect is refused by name, not by code.
        wrong = inputs[0].numpy().astype(np.float64)
        self.assertRaisesRegex(
            RuntimeError,
            "but the method expects",
            executorch_module,
            [wrong, wrong],
        )

    def test_single_buffer_input_is_one_input(self) -> None:
        # A numpy array is also a sequence, so a bare one has to be taken as a
        # single input rather than as one input per row.
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_module = self.load_fn(exported_program.buffer)

        expected = np.asarray(executorch_module(inputs)[0])
        self.assertTrue(
            np.array_equal(
                expected, np.asarray(executorch_module(inputs[0].numpy())[0])
            )
        )

    def test_buffer_input_without_planned_memory(self) -> None:
        # A method exported without planned input memory reads the caller's memory
        # directly, so nothing is copied. The binding holds a reference to the
        # buffer for as long as the runtime may read it, which is what makes that
        # safe, and a change made before execute() is a change the method sees.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        self.assertFalse(method.method_meta().input_tensor_meta(0).is_memory_planned())

        arrays = [i.numpy().copy() for i in inputs]
        method.set_inputs(arrays)
        arrays[0][:] = 9.0
        method.execute()
        expected = arrays[0] + arrays[1]
        self.assertTrue(np.array_equal(expected, np.asarray(method.get_outputs()[0])))

    def test_input_with_a_size_the_runtime_cannot_hold_is_refused(self) -> None:
        # The runtime keeps sizes in a 32 bit signed type, and a larger size wraps
        # to a negative one when it narrows, which fails an assertion deep inside
        # rather than raising. A tensor with no elements can carry such a size
        # without using any memory, so this needs no allocation to reach.
        exported_program, _ = create_program(ModuleAddSingleInput())
        too_large = 2**31 + 5
        for runnable in (
            self.load_fn(exported_program.buffer),
            self.load_prog_fn(exported_program.buffer).load_method("forward"),
        ):
            for described in (
                np.empty((too_large, 0), dtype=np.float32),
                torch.empty((too_large, 0)),
            ):
                with self.assertRaisesRegex(RuntimeError, "outside what the runtime"):
                    runnable([described])

    def test_input_whose_memory_moved_is_refused(self) -> None:
        # Reading the caller's memory in place rests on the buffer protocol keeping
        # that memory still while a view is held. numpy offers an explicit way out
        # of that promise, and taking it leaves the runtime pointing at memory that
        # has been freed, which reads as numbers rather than as a failure. So the
        # promise is checked rather than trusted.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        self.assertFalse(method.method_meta().input_tensor_meta(0).is_memory_planned())

        moving = inputs[0].numpy().copy()
        method.set_inputs([moving, inputs[1].numpy().copy()])
        moving.resize(1024 * 1024, refcheck=False)
        with self.assertRaisesRegex(RuntimeError, "moved after it was set"):
            method.execute()
        # And the method says so again rather than running on what is left.
        with self.assertRaisesRegex(RuntimeError, "Set all of them again|moved after"):
            method.execute()

    def test_dlpack_input_from_a_producer_that_is_nothing_else(self) -> None:
        # DLPack is how array libraries hand each other memory. It describes dtypes
        # the buffer protocol cannot name and memory the host cannot read, which is
        # why it is the way in for an accelerator.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        arrays = [i.numpy().copy() for i in inputs]
        expected = arrays[0] + arrays[1]

        handed = [_OnlyDlpack(a) for a in arrays]
        self.assertTrue(np.array_equal(expected, np.asarray(method(handed)[0])))
        # A claimed capsule is a promise that the memory stays put, so it is read
        # where it is, and a change made before running is a change the method sees.
        method.set_inputs(handed)
        arrays[0][:] = 9.0
        method.execute()
        self.assertTrue(
            np.array_equal(arrays[0] + arrays[1], np.asarray(method.get_outputs()[0]))
        )
        # And a dtype the method did not ask for is still named.
        with self.assertRaisesRegex(RuntimeError, "but the method expects"):
            method([_OnlyDlpack(arrays[0].astype(np.float64)), handed[1]])

    def test_capsule_that_describes_its_memory_dishonestly_is_refused(self) -> None:
        # A capsule is a bare struct, so its rank, its shape pointer and its data
        # pointer are numbers nobody vouches for, unlike every other input, which
        # is an object describing its own memory. Read without checking, a rank of
        # a million reads a million sizes from a two size array and a shape
        # pointer of zero reads from nothing, both of which end the process rather
        # than raise. So the cases run in a child, and one that crashes takes the
        # ones after it with it. Null memory is the other half: every other way in
        # refuses it, and the last two cases are the same refusal for a shape that
        # holds elements and for one whose element count wraps to zero.
        exported_program, _ = create_program(ModuleAdd())
        with tempfile.NamedTemporaryFile(suffix=".pte") as pte:
            pte.write(exported_program.buffer)
            pte.flush()
            child = subprocess.run(
                [sys.executable, "-c", _HOSTILE_CAPSULE, pte.name],
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        honest, rank, shape, data, wrapping, negative = child.stdout.splitlines()
        self.assertEqual(honest, "ran")
        self.assertRegex(rank, r"^RuntimeError .*1048576 dimensions, which is more")
        self.assertRegex(shape, r"^RuntimeError .*hands over no shape")
        self.assertRegex(data, r"^RuntimeError .*has no data")
        self.assertRegex(wrapping, r"^RuntimeError .*has no data")
        self.assertRegex(negative, r"^RuntimeError .*-3 dimensions, which is not")

    def test_refused_set_inputs_keeps_the_capsule_it_claimed(self) -> None:
        # Claiming a capsule promises the producer that the memory stays where it
        # is until the deleter is called. A set_inputs the runtime refuses partway
        # leaves it holding pointers into the inputs it did install, so calling
        # the deleter then hands back memory a method still points at. Numpy's
        # capsule holds a reference to the array it came from, which makes the
        # promise observable: while the capsule is held, the array cannot be
        # collected.
        class OnlyDlpack:
            def __init__(self, array) -> None:
                self._array = array

            def __dlpack__(self, stream=None):
                return self._array.__dlpack__(stream=stream)

            def __dlpack_device__(self):
                return self._array.__dlpack_device__()

        exported_program, _ = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        producer = OnlyDlpack(np.ones((2, 2), np.float32))
        watched = weakref.ref(producer._array)
        # The first input is accepted and installed, the second is refused for
        # its shape, which only the runtime can see.
        with self.assertRaises(RuntimeError):
            method.set_inputs([producer, np.full((3, 3), 9.0, np.float32)])
        del producer
        gc.collect()
        self.assertIsNotNone(watched(), "the capsule was given back too early")

        # A call the runtime accepts replaces every pointer it was holding, and
        # only then is the capsule given back.
        method.set_inputs([np.ones((2, 2), np.float32)] * 2)
        gc.collect()
        self.assertIsNone(watched(), "the claimed capsule was never given back")

    def test_result_can_be_handed_over_through_dlpack(self) -> None:
        # The result lends its memory out through DLPack as well as through the
        # buffer protocol, which is what lets a consumer read a dtype the buffer
        # protocol cannot name, and what will let one read device memory.
        self.require_torch_free()
        exported_program, inputs = create_program(ModuleAdd())
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        result = method([i.numpy().copy() for i in inputs])[0]

        self.assertEqual(result.__dlpack_device__(), (1, 0))  # kDLCPU, device zero
        viewed = np.from_dlpack(result)
        self.assertTrue(np.array_equal((inputs[0] + inputs[1]).numpy(), viewed))
        # The consumer may outlive the result it came from.
        del result
        gc.collect()
        self.assertTrue(np.array_equal((inputs[0] + inputs[1]).numpy(), viewed))

    def test_read_only_buffer_input_is_not_written(self) -> None:
        # A read only buffer may not be lent to a method that keeps the pointer,
        # because a kernel is free to write to memory the runtime was given. Python
        # objects like bytes are immutable and may be shared, so writing into one
        # corrupts whatever else holds it.
        exported_program, inputs = create_program(
            ModuleAddInPlace(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        self.assertFalse(method.method_meta().input_tensor_meta(0).is_memory_planned())

        immutable = inputs[0].numpy().tobytes()
        read_only = np.frombuffer(immutable, dtype=np.float32).reshape(inputs[0].shape)
        self.assertFalse(read_only.flags.writeable)

        method.set_inputs([read_only, inputs[1].numpy().copy()])
        method.execute()
        self.assertEqual(immutable, inputs[0].numpy().tobytes())
        # And the method ran on what the caller passed, not on a copy of something
        # else, so a copy path that reads the wrong memory is not green here.
        self.assertTrue(
            np.array_equal(
                (inputs[0] + inputs[1]).numpy(),
                np.asarray(method.get_outputs()[0]),
            )
        )

    def test_refused_set_inputs_keeps_the_previous_inputs(self) -> None:
        # A refused set_inputs has to leave the previous inputs, and the
        # references that keep their memory alive, exactly as they were. A method
        # that reads the caller's memory directly is still pointing at them, so
        # dropping them early leaves it reading memory nobody holds.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        expected = (inputs[0] + inputs[1]).numpy()

        # The caller keeps no reference of its own, so only the binding holds them.
        method.set_inputs([i.numpy().copy() for i in inputs])
        with self.assertRaises(RuntimeError):
            method.set_inputs([inputs[0].numpy().copy(), "not a tensor"])

        gc.collect()
        # Give the allocator every chance to reuse memory that was handed back.
        churn = [np.full((2, 2), -999.0, np.float32) for _ in range(20000)]
        method.execute()
        self.assertTrue(np.array_equal(expected, np.asarray(method.get_outputs()[0])))
        del churn

    def test_partly_applied_set_inputs_is_refused_not_executed(self) -> None:
        # The runtime installs inputs one at a time, so a refusal partway through
        # leaves a mix of two calls. The memory of both stays alive, and executing
        # on that mix is refused rather than quietly computing something nobody
        # asked for.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        method.set_inputs(
            [np.ones((2, 2), np.float32), np.full((2, 2), 2.0, np.float32)]
        )
        # The first input is accepted and installed, the second is refused for its
        # shape, which only the runtime can see.
        with self.assertRaises(RuntimeError):
            method.set_inputs(
                [np.full((2, 2), 5.0, np.float32), np.full((3, 3), 9.0, np.float32)]
            )
        gc.collect()
        with self.assertRaises(RuntimeError):
            method.execute()

        # Setting all of them again clears it.
        method.set_inputs(
            [np.ones((2, 2), np.float32), np.full((2, 2), 2.0, np.float32)]
        )
        method.execute()
        self.assertTrue(
            np.array_equal(
                np.full((2, 2), 3.0, np.float32),
                np.asarray(method.get_outputs()[0]),
            )
        )

    def test_outputs_are_refused_after_a_refused_call(self) -> None:
        # A refusal drops what the bindings were holding for the inputs, so
        # anything that then reads an output tied to that memory is reading a
        # hole. Every entry point that can be reached afterwards has to say so
        # rather than hand back what is left.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        arrays = [i.numpy().copy() for i in inputs]
        expected = arrays[0] + arrays[1]

        # A good run first, so that there are outputs to be tempted by, and then
        # a set_inputs the runtime only partly applied.
        method.set_inputs(arrays)
        method.execute()
        self.assertTrue(np.array_equal(expected, np.asarray(method.get_outputs()[0])))
        with self.assertRaises(RuntimeError):
            method.set_inputs([arrays[0], np.full((3, 3), 9.0, np.float32)])
        with self.assertRaisesRegex(RuntimeError, "no outputs to hand back"):
            method.get_outputs()

        # And an execute refused because the memory behind an input moved, which
        # is the one that released what it was holding.
        method.set_inputs(arrays)
        method.execute()
        moving = inputs[0].numpy().copy()
        method.set_inputs([moving, arrays[1]])
        moving.resize(1024 * 1024, refcheck=False)
        with self.assertRaisesRegex(RuntimeError, "moved after it was set"):
            method.execute()
        with self.assertRaisesRegex(RuntimeError, "no outputs to hand back"):
            method.get_outputs()

        # The metadata does not depend on the inputs, so it is still readable,
        # and setting all of them again clears the refusal.
        self.assertEqual(method.method_meta().num_inputs(), 2)
        method.set_inputs(arrays)
        method.execute()
        self.assertTrue(np.array_equal(expected, np.asarray(method.get_outputs()[0])))

    def test_reading_outputs_before_an_execution_is_refused(self) -> None:
        # An output the method was exported without planned memory for has no
        # memory at all until an execution gives it some, so reading it before
        # then reads a null pointer and takes the process down rather than
        # raising. Both ways in are checked in a child for that reason: reading
        # the outputs of a method that has not run, and reading them after an
        # execute that was refused because an input moved.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            ),
        )
        with tempfile.NamedTemporaryFile(suffix=".pte") as pte:
            pte.write(exported_program.buffer)
            pte.flush()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        f"""\
                        import numpy as np
                        from executorch.extension.pybindings.portable_lib import (
                            _load_program,
                        )

                        def refused(call):
                            try:
                                call()
                            except RuntimeError:
                                return "refused"
                            return "returned"

                        method = _load_program({pte.name!r}).load_method("forward")
                        print(refused(method.get_outputs))

                        moving = np.ones((2, 2), np.float32)
                        method.set_inputs([moving, np.ones((2, 2), np.float32)])
                        moving.resize(1024 * 1024, refcheck=False)
                        print(refused(method.execute))
                        print(refused(method.get_outputs))
                        """
                    ),
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        self.assertEqual(child.stdout.split(), ["refused"] * 3)

    def test_torch_input_without_planned_memory_is_copied(self) -> None:
        # A buffer pins its memory and a tensor does not, so a tensor handed to a
        # method that keeps the pointer has to be copied. Without the copy a change
        # made after set_inputs reaches the method, which is the visible half of the
        # same problem as reading it after it is freed.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")

        first = inputs[0].clone()
        second = inputs[1].clone()
        expected = (first + second).numpy()
        method.set_inputs([first, second])
        first.fill_(99.0)
        method.execute()
        self.assertTrue(np.array_equal(expected, np.asarray(method.get_outputs()[0])))

    def test_torch_free_build_runs_without_importing_torch(self) -> None:
        # Two things only observable in a process that never imports torch: results
        # come back as the extension's own buffer type, and importing the bindings
        # does not pull torch in. This file imports torch itself, so the check runs in
        # a child.
        self.require_torch_free()

        exported_program, _ = create_program(ModuleAdd())
        with tempfile.NamedTemporaryFile(suffix=".pte") as pte:
            pte.write(exported_program.buffer)
            pte.flush()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys, numpy as np\n"
                    "from executorch.extension.pybindings.portable_lib import "
                    "_load_for_executorch\n"
                    f"m = _load_for_executorch({pte.name!r})\n"
                    "out = m([np.ones((2, 2), np.float32)] * 2)[0]\n"
                    "print(type(out).__name__, float(np.asarray(out)[0][0]), "
                    "'torch' in sys.modules)\n" + _CHECK_TORCH_FREE,
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.split(), ["TensorBuffer", "2.0", "False"])

    def test_result_type_does_not_change_when_torch_is_imported(self) -> None:
        # Whether a result comes back as a torch tensor is asked once and then kept.
        # Asked per call it would change under a caller who imports torch after the
        # object was built. This file imports torch, so the check runs in a child
        # that starts without it.
        self.require_torch_free()

        exported_program, _ = create_program(ModuleAdd())
        with tempfile.NamedTemporaryFile(suffix=".pte") as pte:
            pte.write(exported_program.buffer)
            pte.flush()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        f"""\
                        import sys
                        import numpy as np
                        from executorch.extension.pybindings.portable_lib import (
                            _load_for_executorch,
                            _load_program,
                        )

                        inputs = [np.ones((2, 2), np.float32)] * 2
                        method = _load_program({pte.name!r}).load_method("forward")
                        module = _load_for_executorch({pte.name!r})
                        assert "torch" not in sys.modules, "torch was already here"
                        before = [
                            type(method(inputs)[0]).__name__,
                            type(module(inputs)[0]).__name__,
                        ]

                        import torch  # noqa: F401

                        assert "torch" in sys.modules
                        print(
                            *before,
                            type(method(inputs)[0]).__name__,
                            type(module(inputs)[0]).__name__,
                        )
                        """
                    ),
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        self.assertEqual(child.stdout.split(), ["TensorBuffer"] * 4)

    def test_a_program_loaded_after_torch_returns_torch_tensors(self) -> None:
        # The answer belongs to the process, not to each object that hands results
        # back, so a program loaded before torch was imported and one loaded after
        # it cannot return two different types for the same runtime.
        self.require_torch_free()

        exported_program, _ = create_program(ModuleAdd())
        with tempfile.NamedTemporaryFile(suffix=".pte") as pte:
            pte.write(exported_program.buffer)
            pte.flush()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        f"""\
                        import sys
                        import numpy as np
                        from executorch.extension.pybindings.portable_lib import (
                            _load_for_executorch,
                            _load_program,
                        )

                        inputs = [np.ones((2, 2), np.float32)] * 2
                        module = _load_for_executorch({pte.name!r})
                        assert "torch" not in sys.modules, "torch was already here"

                        before = type(module(inputs)[0]).__name__
                        import torch  # noqa: F401

                        method = _load_program({pte.name!r}).load_method("forward")
                        print(
                            before,
                            type(module(inputs)[0]).__name__,
                            type(method(inputs)[0]).__name__,
                        )
                        """
                    ),
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        before, after, loaded_later = child.stdout.split()
        # An object's result type is fixed when it is built, so a caller's type
        # never changes under them.
        self.assertEqual(before, after)
        self.assertEqual(before, "TensorBuffer")
        # A program loaded once torch is present hands back torch tensors, which is
        # what keeps existing callers working when this build becomes the published one.
        self.assertEqual(loaded_later, "Tensor")

    def test_dynamically_shaped_output_reports_its_own_size(self) -> None:
        # The buffer an unplanned output is written into is sized for the largest shape
        # the method declares, and a run below that maximum fills less of it. Handing
        # the whole buffer over would tell the reader there are more elements than were
        # written. Only the build without torch exposes that buffer, and only in a
        # process that did not import torch, so the check runs in a child.
        self.require_torch_free()

        class Double(torch.nn.Module):
            def forward(self, x):
                return x * 2

        example = (torch.arange(16, dtype=torch.float32).reshape(8, 2),)
        exported = to_edge(
            torch.export.export(
                Double(),
                example,
                dynamic_shapes={"x": {0: torch.export.Dim("rows", min=2, max=8)}},
            )
        ).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".pte") as pte:
            pte.write(exported.buffer)
            pte.flush()
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json\n"
                    "import numpy as np\n"
                    "from executorch.extension.pybindings.portable_lib import "
                    "_load_program\n"
                    f"m = _load_program({pte.name!r}).load_method('forward')\n"
                    "declared = m.method_meta().output_tensor_meta(0).nbytes()\n"
                    "out = m([np.arange(4, dtype=np.float32).reshape(2, 2)])[0]\n"
                    "print(json.dumps([declared, out.nbytes, list(out.shape), "
                    "np.asarray(out).ravel().tolist()]))\n" + _CHECK_TORCH_FREE,
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(child.returncode, 0, child.stderr)
        declared, actual, shape, values = json.loads(child.stdout)
        self.assertEqual(declared, 8 * 2 * 4)
        self.assertEqual(actual, 2 * 2 * 4)
        self.assertEqual(shape, [2, 2])
        self.assertEqual(values, [0.0, 2.0, 4.0, 6.0])

    def test_method_returning_its_input(self) -> None:
        # A method can return one of its inputs unchanged, and then the two are the
        # same value. Giving that output a buffer of its own would move the input
        # into that buffer, so the method would return whatever the buffer held.
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        example = (torch.tensor([1.0, 3.0]),)
        exported = to_edge(torch.export.export(Identity(), example)).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                )
            )
        )
        method = self.load_prog_fn(exported.buffer).load_method("forward")
        self.assertTrue(torch.equal(example[0], method(example)[0]))

    def test_output_without_planned_memory_is_not_reused(self) -> None:
        # An output the method was exported without planned memory for is handed
        # over as the buffer the kernels wrote into, so every call needs its own.
        # Otherwise a later call writes over a result the caller still holds. Only
        # the build without torch hands that buffer over; a build that links torch
        # copies such an output whatever clone_outputs asks for, so there the pair
        # below only shows that the second call computed its own answer.
        exported_program, inputs = create_program(
            ModuleAdd(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            ),
        )
        method = self.load_prog_fn(exported_program.buffer).load_method("forward")
        self.assertFalse(method.method_meta().output_tensor_meta(0).is_memory_planned())

        arrays = [i.numpy().copy() for i in inputs]
        # Asking for no copy is what makes this observable: the result then refers
        # to the buffer the kernels wrote into, so reusing that buffer for the next
        # call would change a result already handed over.
        first = method(arrays, clone_outputs=False)[0]
        kept = np.asarray(first).copy()
        second = method([arrays[0] + 100, arrays[1]], clone_outputs=False)[0]
        self.assertTrue(np.array_equal(kept + 100, np.asarray(second)))
        self.assertTrue(np.array_equal(kept, np.asarray(first)))

    def test_negated_view_input_is_refused(self) -> None:
        # torch applies the sign when something reads the tensor, so the memory
        # underneath a negated view holds different values than the view does.
        # Reading that memory runs the method on the wrong numbers, quietly, so it
        # has to be refused rather than read.
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        for lazy in (torch._neg_view(inputs[0]), torch.conj(inputs[0] * 1j)):
            with self.assertRaisesRegex(RuntimeError, "negated or conjugated"):
                executorch_module([lazy, inputs[1]])

    def test_subclass_of_a_tensor_is_read_as_a_tensor(self) -> None:
        # A subclass of torch.Tensor is a torch tensor, so every check the torch
        # path performs has to apply to it. Recognised by the name its type
        # prints, it is not: the name differs, so it falls through to DLPack,
        # which every torch tensor also offers and which hands the memory over as
        # it is. A negated view holds different values than it reports, so the
        # method then runs on the wrong numbers and nothing says so.
        class MyTensor(torch.Tensor):
            pass

        exported_program, inputs = create_program(ModuleAdd())
        negated = torch._neg_view(inputs[0])
        resolved = negated.resolve_neg()
        expected = (resolved + inputs[1]).numpy()
        for runnable in (
            self.load_fn(exported_program.buffer),
            self.load_prog_fn(exported_program.buffer).load_method("forward"),
        ):
            with self.assertRaisesRegex(RuntimeError, "negated or conjugated"):
                runnable([negated.as_subclass(MyTensor), inputs[1]])
            # Resolved, the negation is in the memory, and a subclass of an
            # ordinary tensor runs like the tensor it is.
            self.assertTrue(
                np.array_equal(
                    expected,
                    np.asarray(
                        runnable([resolved.as_subclass(MyTensor), inputs[1]])[0]
                    ),
                )
            )

    def test_buffer_input_raw_bytes_for_unspellable_dtype(self) -> None:
        # bfloat16 has no buffer format code, and this repository's own example
        # exports default to a bfloat16 input, so the raw bytes of the input have
        # to be a way in. The dtype and the shape then come from the method.
        exported_program, inputs = create_program(ModuleAddDtype(torch.bfloat16))
        executorch_module = self.load_fn(exported_program.buffer)

        expected = executorch_module(inputs)[0]
        raw = [i.view(torch.uint8).numpy().ravel() for i in inputs]
        # numpy cannot name bfloat16 at all, which is why raw bytes are the way in.
        self.assertRaises(TypeError, inputs[0].numpy)
        actual = executorch_module(raw)[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_program_buffer_is_kept_alive(self) -> None:
        # A program is read straight out of the bytes it was loaded from rather than
        # copied, so every loader that takes a buffer has to hold a reference to it.
        # Asked directly, because whether freed bytes still read back correctly
        # depends on the allocator.
        exported_program, inputs = create_program(ModuleAdd())
        buffer = exported_program.buffer
        before = sys.getrefcount(buffer)

        module = self.load_fn(buffer)
        self.assertGreater(sys.getrefcount(buffer), before)

        program = self.load_prog_fn(buffer)
        self.assertGreater(sys.getrefcount(buffer), before + 1)

        # A method keeps them too, since it outlives the program object it came from.
        method = program.load_method("forward")
        del program
        gc.collect()
        self.assertGreater(sys.getrefcount(buffer), before + 1)

        expected = inputs[0] + inputs[1]
        self.assertTrue(torch.equal(expected, module(inputs)[0]))
        self.assertTrue(torch.equal(expected, method(inputs)[0]))

    def test_method_meta(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())

        executorch_module = self.load_fn(exported_program.buffer)
        meta = executorch_module.method_meta("forward")

        del executorch_module
        self.assertEqual(meta.name(), "forward")
        self.assertEqual(meta.num_inputs(), 2)
        self.assertEqual(meta.num_outputs(), 1)

        tensor_info = (
            "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
        )
        float_dtype = 6
        self.assertEqual(
            str(meta),
            "MethodMeta(name='forward', num_inputs=2, "
            f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
            f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
        )

        input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
        output_tensor = meta.output_tensor_meta(0)

        with self.assertRaises(IndexError):
            meta.input_tensor_meta(2)

        del meta
        self.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
        self.assertEqual([t.dtype() for t in input_tensors], [float_dtype, float_dtype])
        self.assertEqual([t.is_memory_planned() for t in input_tensors], [True, True])
        self.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
        self.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

        self.assertEqual(output_tensor.sizes(), (2, 2))
        self.assertEqual(output_tensor.dtype(), float_dtype)
        self.assertEqual(output_tensor.is_memory_planned(), True)
        self.assertEqual(output_tensor.nbytes(), 16)
        self.assertEqual(str(output_tensor), tensor_info)

    def test_bad_name(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError):
            executorch_module.run_method("not_a_real_method", inputs)

    def test_verification_config(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        Verification = self.runtime.Verification

        for config in [Verification.Minimal, Verification.InternalConsistency]:
            executorch_module = self.load_fn(
                exported_program.buffer,
                enable_etdump=False,
                debug_buffer_size=0,
                program_verification=config,
            )

            executorch_output = executorch_module.forward(inputs)[0]
            expected = inputs[0] + inputs[1]
            self.assertEqual(str(expected), str(executorch_output))

    def test_unsupported_input_type(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)
        inputs = ([*inputs],)
        self.assertRaises(RuntimeError, executorch_module, inputs)

    def test_program_methods_one(self):
        exported_program, _ = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)

        self.assertEqual(executorch_program.num_methods(), 1)
        self.assertEqual(executorch_program.get_method_name(0), "forward")

    def test_program_methods_multi(self):
        exported_program, _ = create_program(ModuleMulti())
        executorch_program = self.load_prog_fn(exported_program.buffer)

        self.assertEqual(executorch_program.num_methods(), 2)
        self.assertEqual(executorch_program.get_method_name(0), "forward")
        self.assertEqual(executorch_program.get_method_name(1), "forward2")

    def test_program_method_index_out_of_bounds(self):
        exported_program, _ = create_program(ModuleMulti())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        self.assertRaises(RuntimeError, executorch_program.get_method_name, 2)

    def test_method_e2e(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method.call(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_output_lifespan(self):
        def lower_function_call():
            program, inputs = create_program(ModuleMulti())
            executorch_program = self.load_prog_fn(program.buffer)
            executorch_method = executorch_program.load_method("forward")
            return executorch_method.call(inputs)

        outputs = lower_function_call()
        self.assertTrue(torch.allclose(outputs[0], torch.ones(2, 2) * 2))

    def test_method_multiple_entry(self):
        program, inputs = create_program(ModuleMulti())
        executorch_program = self.load_prog_fn(program.buffer)

        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method.call(inputs)[0]
        self.assertTrue(torch.allclose(executorch_output, torch.ones(2, 2) * 2))

        executorch_method2 = executorch_program.load_method("forward2")
        executorch_output2 = executorch_method2.call(inputs)[0]
        self.assertTrue(torch.allclose(executorch_output2, torch.ones(2, 2) * 3))

    def test_method_by_parts(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")

        executorch_method.set_inputs(inputs)
        executorch_method.execute()
        executorch_output = executorch_method.get_outputs()[0]

        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_callable(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs)[0]
        expected = inputs[0] + inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_single_input(self):
        exported_program, inputs = create_program(ModuleAddSingleInput())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs[0])[0]
        expected = inputs[0] + inputs[0]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_stderr_redirect(self):
        class RedirectedStderr:
            def __init__(self):
                self._stderr = None
                self._string_io = None

            def __enter__(self):
                self._stderr = sys.stderr
                sys.stderr = self._string_io = StringIO()
                return self

            def __exit__(self, type, value, traceback):
                sys.stderr = self._stderr

            def __str__(self):
                return self._string_io.getvalue()

        with RedirectedStderr() as out:
            try:
                program, inputs = create_program(ModuleAdd())
                executorch_program = self.load_prog_fn(program.buffer)
                executorch_method = executorch_program.load_method("forward")
                inputs = (*inputs, 1)
                executorch_output = executorch_method(inputs)[0]  # noqa
                self.assertFalse(True)  # should be unreachable
            except Exception:
                self.assertTrue(str(out).find("The length of given input array"))

    def test_method_quantized_ops(self):
        eager_module = ModuleAdd()

        from executorch.exir import EdgeCompileConfig
        from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.backend_config.executorch import (
            get_executorch_backend_config,
        )
        from torch.ao.quantization.quantize_fx import (
            _convert_to_reference_decomposed_fx,
            prepare_fx,
        )

        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        example_inputs = (
            torch.ones(1, 5, dtype=torch.float32),
            torch.ones(1, 5, dtype=torch.float32),
        )
        m = prepare_fx(
            eager_module,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        m = _convert_to_reference_decomposed_fx(m)
        config = EdgeCompileConfig(_check_ir_validity=False)
        m = to_edge(export(m, example_inputs, strict=True), compile_config=config)
        m = m.transform([QuantFusionPass(_fix_node_meta_val=True)])

        exec_prog = m.to_executorch()

        executorch_program = self.load_prog_fn(exec_prog.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(example_inputs)[0]

        expected = example_inputs[0] + example_inputs[1]
        self.assertEqual(str(expected), str(executorch_output))

    def test_method_constant_output_not_memory_planned(self):
        exported_program, _ = create_program(
            ModuleAddConstReturn(),
            et_config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
            ),
        )

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method((torch.ones(2, 2),))

        expected = torch.ones(2, 2) + torch.ones(2, 2)
        self.assertTrue(torch.allclose(expected, executorch_output[0]))
        self.assertEqual(str(torch.ones(2, 2)), str(executorch_output[1]))

    def test_method_channels_last(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_method_unsupported_dim_order(self) -> None:
        model = ModuleChannelsLast()
        exported_program, inputs = create_program(model)
        inputs = (torch.randn(1, 2, 3, 4, 5).to(memory_format=torch.channels_last_3d),)

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        self.assertRaises(RuntimeError, executorch_method, inputs[0])

    def test_method_channels_last_in_default_out(self) -> None:
        model = ModuleChannelsLastInDefaultOut()
        exported_program, inputs = create_program(model)

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_output = executorch_method(inputs[0])[0]

        expected = model(inputs[0])
        self.assertTrue(torch.allclose(expected, executorch_output))

    def test_method_bad_name(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError):
            executorch_program.load_method("not_a_real_method")

    def test_program_verification_config(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())
        Verification = self.runtime.Verification

        for config in [Verification.Minimal, Verification.InternalConsistency]:
            executorch_program = self.load_prog_fn(
                exported_program.buffer,
                enable_etdump=False,
                debug_buffer_size=0,
                program_verification=config,
            )

            executorch_method = executorch_program.load_method("forward")
            executorch_output = executorch_method(inputs)[0]

            expected = inputs[0] + inputs[1]
            self.assertEqual(str(expected), str(executorch_output))

    def test_method_unsupported_input_type(self):
        exported_program, inputs = create_program(ModuleAdd())
        executorch_program = self.load_prog_fn(exported_program.buffer)
        inputs = ([*inputs],)
        executorch_method = executorch_program.load_method("forward")
        self.assertRaises(RuntimeError, executorch_method, inputs)

    def test_method_attribute(self):
        eager_module = ModuleAddWithAttributes()
        inputs = eager_module.get_inputs()

        exported_program = export(eager_module, inputs, strict=True)
        exec_prog = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                emit_mutable_buffer_names=True,
            )
        )

        exec_prog.dump_executorch_program(verbose=True)

        executorch_program = self.load_prog_fn(exec_prog.buffer)
        executorch_method = executorch_program.load_method("forward")
        executorch_method(inputs)
        self.assertEqual(
            str(executorch_method.get_attribute("state")), str(torch.ones(2, 2))
        )

    def test_program_method_meta(self) -> None:
        eager_module = ModuleAddWithAttributes()
        inputs = eager_module.get_inputs()

        exported_program = export(eager_module, inputs, strict=True)
        exec_prog = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                emit_mutable_buffer_names=True,
            )
        )

        exec_prog.dump_executorch_program(verbose=True)

        executorch_program = self.load_prog_fn(exec_prog.buffer)

        meta = executorch_program.method_meta("forward")

        del executorch_program
        self.assertEqual(meta.name(), "forward")
        self.assertEqual(meta.num_inputs(), 2)
        self.assertEqual(meta.num_outputs(), 1)
        self.assertEqual(meta.num_attributes(), 1)

        tensor_info = (
            "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
        )

        float_dtype = 6
        self.assertEqual(
            str(meta),
            "MethodMeta(name='forward', num_inputs=2, "
            f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
            f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
        )

        input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
        output_tensor = meta.output_tensor_meta(0)
        attribute_tensor = meta.attribute_tensor_meta(0)

        with self.assertRaises(IndexError):
            meta.input_tensor_meta(2)

        with self.assertRaises(IndexError):
            meta.attribute_tensor_meta(1)

        del meta
        self.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
        self.assertEqual([t.dtype() for t in input_tensors], [float_dtype, float_dtype])
        self.assertEqual([t.is_memory_planned() for t in input_tensors], [True, True])
        self.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
        self.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

        self.assertEqual(output_tensor.sizes(), (2, 2))
        self.assertEqual(output_tensor.dtype(), float_dtype)
        self.assertEqual(output_tensor.is_memory_planned(), True)
        self.assertEqual(output_tensor.nbytes(), 16)
        self.assertEqual(str(output_tensor), tensor_info)

        self.assertEqual(attribute_tensor.sizes(), (2, 2))
        self.assertEqual(attribute_tensor.dtype(), float_dtype)
        self.assertEqual(attribute_tensor.is_memory_planned(), True)
        self.assertEqual(attribute_tensor.nbytes(), 16)
        self.assertEqual(str(attribute_tensor), tensor_info)

    def test_method_method_meta(self) -> None:
        exported_program, inputs = create_program(ModuleAdd())

        executorch_program = self.load_prog_fn(exported_program.buffer)
        executorch_method = executorch_program.load_method("forward")
        meta = executorch_method.method_meta()

        del executorch_program
        del executorch_method
        self.assertEqual(meta.name(), "forward")
        self.assertEqual(meta.num_inputs(), 2)
        self.assertEqual(meta.num_outputs(), 1)

        tensor_info = (
            "TensorInfo(sizes=[2, 2], dtype=Float, is_memory_planned=True, nbytes=16)"
        )
        float_dtype = 6
        self.assertEqual(
            str(meta),
            "MethodMeta(name='forward', num_inputs=2, "
            f"input_tensor_meta=['{tensor_info}', '{tensor_info}'], "
            f"num_outputs=1, output_tensor_meta=['{tensor_info}'])",
        )

        input_tensors = [meta.input_tensor_meta(i) for i in range(2)]
        output_tensor = meta.output_tensor_meta(0)

        with self.assertRaises(IndexError):
            meta.input_tensor_meta(2)

        del meta
        self.assertEqual([t.sizes() for t in input_tensors], [(2, 2), (2, 2)])
        self.assertEqual([t.dtype() for t in input_tensors], [float_dtype, float_dtype])
        self.assertEqual([t.is_memory_planned() for t in input_tensors], [True, True])
        self.assertEqual([t.nbytes() for t in input_tensors], [16, 16])
        self.assertEqual(str(input_tensors), f"[{tensor_info}, {tensor_info}]")

        self.assertEqual(output_tensor.sizes(), (2, 2))
        self.assertEqual(output_tensor.dtype(), float_dtype)
        self.assertEqual(output_tensor.is_memory_planned(), True)
        self.assertEqual(output_tensor.nbytes(), 16)
        self.assertEqual(str(output_tensor), tensor_info)

    def test_program_data_separation(self) -> None:
        eager_module = ModuleLinear()
        inputs = eager_module.get_inputs()
        exported_program = export(eager_module, inputs, strict=True)
        exec_program = to_edge(exported_program).to_executorch(
            config=ExecutorchBackendConfig(
                # Move all tensor data to '_default_external_constant' file.
                external_constants=True,
            )
        )
        program_buffer = exec_program.buffer
        assert len(exec_program._tensor_data) == 1
        data_buffer = bytes(exec_program._tensor_data.pop("_default_external_constant"))

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pte_file = os.path.join(tmpdir, "linear.pte")
            with open(pte_file, "wb") as f:
                f.write(program_buffer)
            ptd_file = os.path.join(tmpdir, "linear.ptd")
            with open(ptd_file, "wb") as ptd:
                ptd.write(data_buffer)
            expected = eager_module(inputs[0])
            # Test 1: File-based loading with external data file
            executorch_module_file = self.runtime._load_for_executorch(
                pte_file, ptd_file
            )
            executorch_output_file = executorch_module_file.forward(inputs)[0]
            self.assertTrue(torch.allclose(expected, executorch_output_file))

        # Test 2: Buffer-based loading with external data buffer
        executorch_module_buffer = self.load_fn(program_buffer, data_buffer)
        executorch_output_buffer = executorch_module_buffer.forward(inputs)[0]
        self.assertTrue(torch.allclose(expected, executorch_output_buffer))

        # Test 3: Buffer-based loading without external data file (should fail or work differently)
        # This should fail because the program expects external data
        executorch_module_no_data = self.load_fn(program_buffer)
        with self.assertRaises(RuntimeError):
            executorch_module_no_data.forward(inputs)

        # Test 4: Test with invalid data buffer (should fail)
        invalid_bytes = b"invalid bytes"
        executorch_module_invalid_data = self.load_fn(program_buffer, invalid_bytes)
        with self.assertRaises(RuntimeError):
            executorch_module_invalid_data.forward(inputs)

        # Test 5: Test bundled program loading with external data
        # First create a bundled program with external constants
        from executorch.devtools.bundled_program.config import (
            MethodTestCase,
            MethodTestSuite,
        )
        from executorch.devtools.bundled_program.core import BundledProgram
        from executorch.devtools.bundled_program.serialize import (
            serialize_from_bundled_program_to_flatbuffer,
        )

        method_test_suites = [
            MethodTestSuite(
                method_name="forward",
                test_cases=[
                    MethodTestCase(
                        inputs=input,
                        expected_outputs=expected,
                    )
                    for input in inputs
                ],
            ),
        ]
        bundled_program = BundledProgram(exec_program, method_test_suites)
        bundled_buffer = serialize_from_bundled_program_to_flatbuffer(bundled_program)
        bundled_module = self.runtime._load_bundled_program_from_buffer(bundled_buffer)

        # Load module from bundled program with external data buffer
        executorch_module_bundled = (
            self.runtime._load_for_executorch_from_bundled_program(
                bundled_module, data_buffer
            )
        )
        executorch_output_bundled = executorch_module_bundled.forward(inputs)[0]
        self.assertTrue(torch.allclose(expected, executorch_output_bundled))

        # Load module from bundled program with external data file
        with tempfile.TemporaryDirectory() as tmpdir:
            ptd_file = os.path.join(tmpdir, "linear.ptd")
            with open(ptd_file, "wb") as ptd:
                ptd.write(data_buffer)
            executorch_module_bundled_data_file = (
                self.runtime._load_for_executorch_from_bundled_program(
                    bundled_module, ptd_file
                )
            )
            executorch_output_bundled_data_file = (
                executorch_module_bundled_data_file.forward(inputs)[0]
            )
            self.assertTrue(
                torch.allclose(expected, executorch_output_bundled_data_file)
            )

        # Test 6: Bundled program without external data should fail
        executorch_module_bundled_no_data = (
            self.runtime._load_for_executorch_from_bundled_program(bundled_module)
        )
        with self.assertRaises(RuntimeError):
            executorch_module_bundled_no_data.forward(inputs)

    def test_method_rejects_input_on_unrepresentable_device(self):
        # The conversion used to label every input CPU, so an input whose memory
        # the host cannot read was described as host memory, and the failure
        # surfaced later as a crash instead of an error naming the input.
        if self.kernel_mode != "portable":
            self.skipTest("only the portable build converts the input tensor")

        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        with self.assertRaises(RuntimeError) as caught:
            executorch_module.forward([inputs[0].to("meta"), inputs[1]])
        message = str(caught.exception)
        # Asserts the device is named as Python spells it, since an uppercased or
        # index-less name would not match what the caller passed.
        self.assertIn("is on device meta", message)
        self.assertIn("only CPU and CUDA tensors", message)

    def test_method_accepts_a_cpu_input_after_the_device_check(self):
        # The rejection tests above pass for a change that throws on every input, so this
        # asserts the other half: an ordinary CPU input still converts and runs. Without it
        # the pair does not distinguish "rejects what it cannot represent" from "rejects
        # everything".
        exported_program, inputs = create_program(ModuleAdd())
        executorch_module = self.load_fn(exported_program.buffer)

        executorch_output = executorch_module.forward(inputs)[0]

        self.assertTrue(
            torch.allclose(executorch_output, inputs[0] + inputs[1]),
            "a CPU input must still reach the method and produce the same result",
        )

    def test_program_method_rejects_input_on_unrepresentable_device(self):
        # The other conversion site, reached through a loaded program rather than a
        # module. It got the same treatment, and without this it had no test: before
        # the change this path aborted the process rather than raising.
        if self.kernel_mode != "portable":
            self.skipTest("only the portable build converts the input tensor")

        exported_program, inputs = create_program(ModuleAdd())
        program = self.load_prog_fn(exported_program.buffer)
        method = program.load_method("forward")

        with self.assertRaises(RuntimeError) as caught:
            method.set_inputs([inputs[0].to("meta"), inputs[1]])
        message = str(caught.exception)
        self.assertIn("is on device meta", message)
        self.assertIn("only CPU and CUDA tensors", message)

    def test_program_loads_when_one_method_is_device_planned(self):
        # Linking the CUDA backend registers a CUDA allocator at static init, and the
        # registry has no way to drop one, so there the device load would succeed with
        # or without the fix and this test could not tell them apart.
        if "CudaBackend" in self.runtime._get_registered_backend_names():
            self.skipTest("a registered CUDA allocator satisfies the device load")

        exported_program, inputs = create_program(
            ModuleMulti(),
            et_config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
            partitioner={"forward2": DeviceAwarePartitioner()},
        )

        # Without this the test would quietly become a second copy of
        # test_method_multiple_entry if the planner stopped tagging devices.
        planned_devices = {
            plan.name: [
                buffer.device_type for buffer in (plan.non_const_buffer_device or [])
            ]
            for plan in exported_program.executorch_program.execution_plan
        }
        self.assertIn(DeviceType.CUDA, planned_devices["forward2"])
        self.assertNotIn(DeviceType.CUDA, planned_devices["forward"])

        program = self.load_prog_fn(exported_program.buffer)
        self.assertEqual(program.num_methods(), 2)

        method = program.load_method("forward")
        self.assertTrue(torch.allclose(method.call(inputs)[0], torch.ones(2, 2) * 2))

        # Asking for device memory at all is what this asserts. Before the fix the
        # loader put every planned buffer in host memory, so the device-planned method
        # never asked and simply loaded. Now it asks, and with no device allocator
        # registered the request is refused.
        with self.assertRaises(RuntimeError) as caught:
            program.load_method("forward2")
        self.assertIn("on device", str(caught.exception))

        # A failed load must leave the method that already loaded usable.
        self.assertTrue(torch.allclose(method.call(inputs)[0], torch.ones(2, 2) * 2))

    def test_device_planned_method_allocates_on_the_device(self):
        # The other device test covers the refusal. This one covers what the
        # refusal is protecting: on a build that does have a device allocator,
        # the arena has to come off the device, not out of host memory. It needs
        # a real accelerator, so it only runs where one is present.
        if "CudaBackend" not in self.runtime._get_registered_backend_names():
            self.skipTest("needs a build with the CUDA backend linked in")
        if not torch.cuda.is_available():
            self.skipTest("needs a visible CUDA device")

        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        # Large enough that the arena is far bigger than the noise in
        # mem_get_info, which moves by a few MiB as contexts are created.
        side = 2048
        inputs = (torch.ones(side, side), torch.ones(side, side))

        class HostOnly(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Delegated(torch.nn.Module):
            def forward(self, x, y):
                return (x + y) * 2.0

        edge = to_edge_transform_and_lower(
            {
                "forward": export(HostOnly(), inputs, strict=True),
                "forward2": export(Delegated(), inputs, strict=True),
            },
            partitioner={
                "forward2": [CudaPartitioner([CompileSpec("method_name", b"forward2")])]
            },
        )
        exported_program = edge.to_executorch(
            config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=True)
        )

        plans = {
            plan.name: plan
            for plan in exported_program.executorch_program.execution_plan
        }
        device_buffers = {
            buffer.buffer_idx
            for buffer in (plans["forward2"].non_const_buffer_device or [])
        }
        self.assertTrue(device_buffers, "the planner tagged nothing for the device")
        self.assertFalse(plans["forward"].non_const_buffer_device or [])
        device_bytes = sum(
            size
            for index, size in enumerate(plans["forward2"].non_const_buffer_sizes)
            if index in device_buffers
        )

        # The CUDA backend keeps its weights in a separate file, so the program
        # has to be loaded from disk with that file alongside it.
        with tempfile.TemporaryDirectory() as directory:
            pte_path = os.path.join(directory, "program.pte")
            with open(pte_path, "wb") as pte_file:
                exported_program.write_to_file(pte_file)
            data_names = sorted(exported_program._tensor_data or {})
            exported_program.write_tensor_data_to_file(directory)
            data_path = (
                os.path.join(directory, data_names[0] + ".ptd") if data_names else None
            )

            torch.cuda.init()
            program = self.runtime._load_program(pte_path, data_path=data_path)

            torch.cuda.synchronize()
            free_before, _ = torch.cuda.mem_get_info()

            # The host-only method must not touch the device at all.
            host_method = program.load_method("forward")
            torch.cuda.synchronize()
            free_after_host, _ = torch.cuda.mem_get_info()
            self.assertLess(free_before - free_after_host, device_bytes // 2)

            device_method = program.load_method("forward2")
            torch.cuda.synchronize()
            free_after_device, _ = torch.cuda.mem_get_info()
            self.assertGreaterEqual(
                free_after_host - free_after_device, device_bytes * 0.9
            )

            # Before the fix this ran on a host pointer and the backend rejected
            # it, so getting the right answer back is itself part of the check.
            expected = (inputs[0] + inputs[1]) * 2.0
            self.assertTrue(
                torch.allclose(device_method.call(inputs)[0].cpu(), expected)
            )
            self.assertTrue(
                torch.allclose(host_method.call(inputs)[0].cpu(), inputs[0] + inputs[1])
            )
            # The device arenas are private, so running the host method in
            # between must not disturb them.
            self.assertTrue(
                torch.allclose(device_method.call(inputs)[0].cpu(), expected)
            )

            del device_method
            del host_method
            del program
            torch.cuda.synchronize()
            free_at_end, _ = torch.cuda.mem_get_info()
            self.assertGreaterEqual(free_at_end - free_after_device, device_bytes * 0.9)
