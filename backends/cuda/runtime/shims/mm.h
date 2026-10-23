/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Matrix multiplication on cuBLASLt, the AOTI shim for aten::mm.out. Inductor
 * calls it when autotuning picked cuBLAS over a Triton template.
 *
 * Every tensor is a 2-D float32, float16 or bfloat16 CUDA tensor of one dtype,
 * and each matrix must be row-major or column-major (a leading dimension may
 * exceed the matrix width, as for a slice). float32 runs in full precision.
 *
 * @param out Result, M x N, written in place
 * @param self Left operand, M x K
 * @param mat2 Right operand, K x N
 * @return Error::Ok on success, Error::InvalidArgument or Error::NotSupported
 * for operands this shim does not handle, Error::Internal on a cuBLAS failure
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_mm_out(Tensor* out, Tensor* self, Tensor* mat2);

/**
 * out = beta * self + alpha * (mat1 @ mat2), the AOTI shim for aten::addmm.out,
 * with the same operand rules as aoti_torch_cuda_mm_out.
 *
 * self is either a bias row of N values broadcast over the M rows (a 1-D
 * tensor, or 2-D with a zero row stride), which needs beta == 1 and a
 * row-major out and runs as a cuBLASLt bias epilogue, or a full M x N matrix
 * laid out like out.
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_addmm_out(
    Tensor* out,
    Tensor* self,
    Tensor* mat1,
    Tensor* mat2,
    double beta,
    double alpha);

/**
 * Batched matrix multiplication, the AOTI shim for aten::bmm.out, with the
 * same operand rules as aoti_torch_cuda_mm_out applied to each matrix of the
 * batch. The tensors are 3-D; an operand may repeat one matrix over the batch
 * with a zero batch stride.
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_bmm_out(Tensor* out, Tensor* self, Tensor* mat2);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
