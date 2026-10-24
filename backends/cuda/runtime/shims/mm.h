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
 * calls it when it uses the ATen choice for a matrix multiplication.
 *
 * Every tensor is a 2-D float32, float16 or bfloat16 CUDA tensor of one dtype,
 * and each matrix must be row-major or column-major (the leading dimension may
 * exceed the row or column length, as for a slice). float32 runs in full
 * precision, not TF32. cuBLASLt is loaded the first time one of these shims
 * runs.
 *
 * @param out Result, M x N, written in place
 * @param self Left operand, M x K
 * @param mat2 Right operand, K x N
 * @return Error::Ok on success, Error::InvalidArgument or Error::NotSupported
 * for operands this shim does not handle, Error::NotFound if cuBLASLt cannot
 * be loaded, Error::Internal on a CUDA or cuBLAS failure, or the error of a
 * failed stream lookup or workspace allocation
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_mm_out(Tensor* out, Tensor* self, Tensor* mat2);

/**
 * out = beta * self + alpha * (mat1 @ mat2), the AOTI shim for aten::addmm.out,
 * with the same operand rules as aoti_torch_cuda_mm_out for mat1, mat2 and
 * out.
 *
 * @param out Result, M x N, written in place
 * @param self Tensor of up to two dimensions that broadcasts to M x N, in any
 * layout. As in PyTorch, it is not read when beta is 0.
 * @param mat1 Left operand, M x K
 * @param mat2 Right operand, K x N
 * @param beta Scale of self
 * @param alpha Scale of mat1 @ mat2
 * @return As for aoti_torch_cuda_mm_out
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
 * batch. An operand may repeat one matrix over the batch with a zero batch
 * stride.
 *
 * @param out Result, B x M x N, written in place
 * @param self Left operand, B x M x K
 * @param mat2 Right operand, B x K x N
 * @return As for aoti_torch_cuda_mm_out
 */
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_bmm_out(Tensor* out, Tensor* self, Tensor* mat2);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
