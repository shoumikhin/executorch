# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Matrix multiplications on cuBLAS through the mm, addmm and bmm CUDA shims.

With max_autotune off, Inductor calls those shims instead of compiling Triton
matrix multiplications into the model.

Usage:
  python -m pytest backends/cuda/tests/test_cublas_gemm.py -v
"""

import glob
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes import MemoryPlanningPass
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu

_CUBLAS_SHIMS = {
    "aoti_torch_cuda_mm_out",
    "aoti_torch_cuda_addmm_out",
    "aoti_torch_cuda_bmm_out",
}


class GemmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.with_bias = torch.nn.Linear(64, 48)
        self.without_bias = torch.nn.Linear(48, 32, bias=False)

    def forward(self, x, y):
        h = self.without_bias(self.with_bias(x))
        return torch.bmm(h.view(4, 4, 32), y)


class AddmmModel(torch.nn.Module):
    def __init__(self, beta, alpha, transpose_self):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.transpose_self = transpose_self

    def forward(self, s, x, w):
        s = s.t() if self.transpose_self else s
        return torch.addmm(s, x, w, beta=self.beta, alpha=self.alpha)


def _export_and_run(model, inputs, max_autotune=True):
    """Lowers `model` and runs it on the ExecuTorch runtime. Returns the
    outputs and the extern shims called."""
    specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    if not max_autotune:
        specs.append(CompileSpec("max_autotune", b"OFF"))
    called = set()
    original = CppWrapperCpu.generate_c_shim_extern_kernel_call

    def record(self, kernel, *args, **kwargs):
        called.add(kernel)
        return original(self, kernel, *args, **kwargs)

    with mock.patch.object(
        CppWrapperCpu, "generate_c_shim_extern_kernel_call", record
    ), torch.no_grad():
        program = to_edge_transform_and_lower(
            torch.export.export(model, inputs, strict=True),
            partitioner=[CudaPartitioner(specs)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True
            ),
        ).to_executorch(
            config=ExecutorchBackendConfig(
                extract_delegate_segments=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            )
        )

    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    with tempfile.TemporaryDirectory() as tmpdir:
        pte = os.path.join(tmpdir, "model.pte")
        with open(pte, "wb") as f:
            program.write_to_file(f)
        program.write_tensor_data_to_file(tmpdir)
        ptd = glob.glob(os.path.join(tmpdir, "*.ptd"))[0]
        module = _load_for_executorch(pte, data_path=ptd)
        outputs = module.run_method("forward", [t.cpu() for t in inputs])
    return outputs, called


class TestCublasGemm(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if torch.version.hip is not None:
            self.skipTest("cuBLAS shims are CUDA only")

    def _check(self, dtype, atol, rtol, **lowering):
        torch.manual_seed(0)
        model = GemmModel().to(device="cuda", dtype=dtype).eval()
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=dtype),
            torch.randn(4, 32, 8, device="cuda", dtype=dtype),
        )
        with torch.no_grad():
            expected = model(*inputs).cpu()

        outputs, called = _export_and_run(model, inputs, **lowering)

        torch.testing.assert_close(outputs[0], expected, atol=atol, rtol=rtol)
        return called

    def test_max_autotune_off(self):
        # Inductor does not autotune and calls the ATen choice for every
        # matrix multiplication.
        called = self._check(torch.float32, 1e-4, 1e-4, max_autotune=False)
        self.assertTrue(_CUBLAS_SHIMS <= called, called)

    def test_addmm_self(self):
        # self of addmm forms other than a linear layer's bias.
        cases = {
            "scaled bias row": (torch.randn(32), 0.5, 1.0, False),
            "column bias": (torch.randn(16, 1), 1.0, 1.0, False),
            "transposed self": (torch.randn(32, 16), 2.0, 0.5, True),
        }
        for name, (s, beta, alpha, transpose_self) in cases.items():
            with self.subTest(name):
                torch.manual_seed(0)
                model = AddmmModel(beta, alpha, transpose_self).eval()
                inputs = (
                    s.cuda(),
                    torch.randn(16, 64, device="cuda"),
                    torch.randn(64, 32, device="cuda"),
                )
                with torch.no_grad():
                    expected = model(*inputs).cpu()

                outputs, called = _export_and_run(model, inputs, max_autotune=False)

                self.assertIn("aoti_torch_cuda_addmm_out", called)
                torch.testing.assert_close(outputs[0], expected, atol=1e-4, rtol=1e-4)


def _archive_symbols(path):
    """The symbols an ar archive's index lists."""
    data = Path(path).read_bytes()
    size = int(data[56:66])
    index = data[68 : 68 + size]
    count = int.from_bytes(index[:4], "big")
    return {name.decode() for name in index[4 + 4 * count :].split(b"\0")[:count]}


class TestWindowsImportLibrary(unittest.TestCase):
    def test_lists_every_cuda_shim(self):
        # Lowering for Windows links the generated code against this library,
        # so a shim it does not list fails to link there.
        library = Path(__file__).parents[1] / "runtime" / "aoti_cuda_shims.lib"
        with mock.patch.object(torch.version, "hip", None):
            shims = {
                name
                for name in CudaBackend.get_supported_fallback_kernels()
                if name.startswith("aoti_torch_cuda_")
            }
        self.assertEqual(shims - _archive_symbols(library), set())


if __name__ == "__main__":
    unittest.main()
