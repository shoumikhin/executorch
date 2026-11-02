# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import unittest

import coremltools as ct

import executorch.exir

import torch
from coremltools.converters.mil.mil import types

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.backend.utils import format_delegated_graph

from torchao.prototype.quantization.codebook_coreml import CodebookWeightOnlyConfig
from torchao.quantization import IntxWeightOnlyConfig, PerAxis, PerGroup, quantize_

if IS_VALID_TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestTorchOps(unittest.TestCase):
    edge_compile_config = executorch.exir.EdgeCompileConfig()

    def _coreml_partitioner(self, *, minimum_deployment_target=ct.target.iOS18):
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=minimum_deployment_target
        )
        return CoreMLPartitioner(compile_specs=compile_specs)

    def _get_test_model(self):
        model = torch.nn.Sequential(
            torch.nn.Embedding(64, 128), torch.nn.Linear(128, 256), torch.nn.ReLU()
        )
        example_inputs = (torch.LongTensor([0]),)
        return model, example_inputs

    def _compare_outputs(
        self, executorch_program, eager_program, example_inputs, atol=1e-2, rtol=1e-2
    ):
        if not IS_VALID_TEST_RUNTIME:
            return
        runtime = Runtime.get()
        program = runtime.load_program(executorch_program.buffer)
        method = program.load_method("forward")
        et_outputs = method.execute(example_inputs)[0]
        eager_outputs = eager_program(*example_inputs)
        self.assertTrue(torch.allclose(et_outputs, eager_outputs, atol=atol, rtol=rtol))

    def test_dequantize_affine_b4w_embedding(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_b4w_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c4w_embedding(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c4w_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0))
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c8w_embedding_b4w_linear(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_affine_c8w_embedding_c8w_linear_ios16(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
        )
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[
                self._coreml_partitioner(minimum_deployment_target=ct.target.iOS16)
            ],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_linear_per_grouped_col(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint2, block_size=[-1, 16]),
        )
        ep = torch.export.export(model, example_inputs)
        assert "torch.ops.quant.dequantize_codebook.default" in ep.graph_module.code
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        assert (
            "executorch.exir.dialects.edge._ops.quant.dequantize_codebook.default"
            in format_delegated_graph(delegated_program.exported_program().graph_module)
        )

        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_linear_per_grouped_row(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint2, block_size=[16, -1]),
        )
        ep = torch.export.export(model, example_inputs)
        assert "torch.ops.quant.dequantize_codebook.default" in ep.graph_module.code
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        assert (
            "executorch.exir.dialects.edge._ops.quant.dequantize_codebook.default"
            in format_delegated_graph(delegated_program.exported_program().graph_module)
        )

        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_embedding_per_grouped_col(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint3, block_size=[-1, 16]),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        assert "torch.ops.quant.dequantize_codebook.default" in ep.graph_module.code
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        assert (
            "executorch.exir.dialects.edge._ops.quant.dequantize_codebook.default"
            in format_delegated_graph(delegated_program.exported_program().graph_module)
        )

        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_dequantize_codebook_embedding_per_grouped_row(self):
        model, example_inputs = self._get_test_model()
        quantize_(
            model,
            CodebookWeightOnlyConfig(dtype=torch.uint3, block_size=[16, -1]),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        ep = torch.export.export(model, example_inputs)
        assert "torch.ops.quant.dequantize_codebook.default" in ep.graph_module.code
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"

        assert (
            "executorch.exir.dialects.edge._ops.quant.dequantize_codebook.default"
            in format_delegated_graph(delegated_program.exported_program().graph_module)
        )

        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test__clone_dim_order_contiguous(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.dim_order_ops._clone_dim_order(
                    x, dim_order=[0, 1, 2, 3]
                )

        model, example_inputs = Model(), (torch.randn(1, 3, 8, 8),)
        ep = torch.export.export(model, example_inputs)
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[self._coreml_partitioner()],
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "executorch_call_delegate",
                    "getitem",
                ], f"Got unexpected node target after delegation: {node.target.__name__}"
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs)

    def test_gelu_tanh(self):
        """
        torch.export passes approximate="tanh" to gelu as a keyword, and
        coremltools reads it only from the positional inputs.  If the keyword
        is dropped, the exact gelu runs, which differs by up to 4.7e-4.
        """
        model = torch.nn.GELU(approximate="tanh").eval()
        example_inputs = (torch.linspace(-8, 8, 4096),)
        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32
        )
        ep = torch.export.export(model, example_inputs)
        program = ct.convert(ep.run_decompositions({}), convert_to="milinternal")
        self.assertEqual(
            [op.mode.val for op in program.find_ops(op_type="gelu")],
            ["TANH_APPROXIMATION"],
        )
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner(compile_specs=compile_specs)]
        )
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                self.assertIn(
                    node.target.__name__, ["executorch_call_delegate", "getitem"]
                )
        et_prog = delegated_program.to_executorch()
        self._compare_outputs(et_prog, model, example_inputs, atol=1e-5, rtol=1e-5)

    def test_fills_keep_their_dtype(self):
        """
        coremltools makes full without a dtype, zeros and new_zeros as float,
        ones_like as float from iOS 16, zeros_like with the dtype of its input,
        and full_like, new_full, fill and scalar_tensor with the type of their
        fill value, instead of the dtype torch gives them. nn.MultiheadAttention
        turns a bool mask into a float one with zeros_like(mask, dtype=...).
        """

        class Fills(torch.nn.Module):
            def forward(self, ids, half):
                ones = torch.ones_like(ids) + ids.new_zeros((2, 3))
                twos = ids.new_full((2, 3), 2.0) + torch.empty_like(ids).fill_(0.0)
                zeros = torch.zeros((2, 3), dtype=torch.long) + torch.zeros_like(ids)
                count = torch.scalar_tensor(1001.0, dtype=torch.long)
                ints = (ids + ones + twos + zeros + torch.full((2, 3), 1)) * count
                flags = torch.where(ids > 0, torch.full((2, 3), True), ids > 1)
                return (
                    ints,
                    flags,
                    half * torch.ones_like(half) + torch.zeros_like(half),
                )

        class Attention(torch.nn.MultiheadAttention):
            def forward(self, x, mask):
                return super().forward(x, x, x, attn_mask=mask, need_weights=False)[0]

        fills = torch.export.export(
            Fills(), (torch.arange(6).reshape(2, 3), torch.rand(2, 3).half())
        )
        program = ct.convert(
            fills.run_decompositions({}),
            convert_to="milinternal",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
        self.assertEqual(
            [output.dtype for output in program.functions["main"].outputs],
            [types.int32, types.bool, types.fp16],
        )
        model = Attention(16, 2, batch_first=True).eval()
        example_inputs = (
            torch.randn(1, 6, 16),
            torch.ones(6, 6, dtype=torch.bool).triu(1),
        )
        attention = torch.export.export(model, example_inputs)
        program = ct.convert(attention.run_decompositions({}), convert_to="milinternal")
        masks = [
            op.outputs[0].dtype
            for op in program.functions["main"].operations
            if op.name.startswith("zeros_like")
        ]
        self.assertTrue(masks and all(types.is_float(dtype) for dtype in masks))
        delegated_program = executorch.exir.to_edge_transform_and_lower(
            attention, partitioner=[CoreMLPartitioner()]
        )
        self.assertEqual(
            [
                node.target.__name__
                for node in delegated_program.exported_program().graph.nodes
                if node.op == "call_function"
            ],
            ["executorch_call_delegate", "getitem"],
        )
        self._compare_outputs(delegated_program.to_executorch(), model, example_inputs)

    def test_dequantize_affine_below_ios18_raises_with_hint(self):
        """
        Regression test for https://github.com/pytorch/executorch/issues/13122.

        `quantize_(...)` with blockwise / int4 configurations requires iOS18.
        coremltools raises a ValueError that does not mention how to fix the
        deployment target on the ExecuTorch side; we wrap it to add the
        partitioner-level guidance.
        """
        model = torch.nn.Linear(64, 64)
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        )
        ep = torch.export.export(model.eval(), (torch.randn(1, 64),), strict=True)
        with self.assertRaises(ValueError) as cm:
            executorch.exir.to_edge_transform_and_lower(
                ep,
                partitioner=[
                    self._coreml_partitioner(minimum_deployment_target=ct.target.iOS17)
                ],
            )
        msg = str(cm.exception)
        self.assertIn("iOS18", msg)
        self.assertIn("CoreMLPartitioner", msg)
        self.assertIn("minimum_deployment_target", msg)


if __name__ == "__main__":
    test_runner = TestTorchOps()
    test_runner.test_dequantize_affine_b4w_embedding()
    test_runner.test_dequantize_affine_b4w_linear()
    test_runner.test_dequantize_affine_c4w_embedding()
    test_runner.test_dequantize_affine_c4w_linear()
    test_runner.test_dequantize_affine_c8w_embedding_b4w_linear()
    test_runner.test_dequantize_affine_c8w_embedding_c8w_linear_ios16()
    test_runner.test_dequantize_codebook_linear_per_grouped_col()
    test_runner.test_dequantize_codebook_linear_per_grouped_row()
    test_runner.test_dequantize_codebook_embedding_per_grouped_col()
    test_runner.test_dequantize_codebook_embedding_per_grouped_row()
    test_runner.test__clone_dim_order_contiguous()
