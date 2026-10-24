/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/mm.h>
#include <executorch/runtime/platform/platform.h>

#include <cmath>
#include <vector>

using namespace executorch::backends::cuda;
using executorch::runtime::Error;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

namespace {

// Small integers, so every product and sum below is exact in float16 and
// bfloat16 too, and results can be compared exactly.
float pattern(int64_t i) {
  return static_cast<float>((i * 7) % 5 - 2);
}

class AOTITorchCudaMmTest
    : public ::testing::TestWithParam<slim_c10::ScalarType> {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  void TearDown() override {
    clearCurrentCUDAStream(0);
    for (Tensor* t : tensors_) {
      aoti_torch_delete_tensor_object(t);
    }
  }

  slim_c10::ScalarType dtype() const {
    return GetParam();
  }

  // A CUDA buffer of `numel` elements holding pattern(i) at element i.
  Tensor* buffer(int64_t numel) {
    Tensor* t = nullptr;
    EXPECT_EQ(
        aoti_torch_empty_strided(
            1,
            &numel,
            nullptr,
            static_cast<int32_t>(dtype()),
            static_cast<int32_t>(slim_c10::DeviceType::CUDA),
            0,
            &t),
        Error::Ok);
    tensors_.push_back(t);
    std::vector<float> values(numel);
    for (int64_t i = 0; i < numel; ++i) {
      values[i] = pattern(i);
    }
    upload(t, values);
    return t;
  }

  Tensor* view(
      Tensor* base,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides) {
    Tensor* t = nullptr;
    EXPECT_EQ(
        aoti_torch__reinterpret_tensor(
            base, sizes.size(), sizes.data(), strides.data(), 0, &t),
        Error::Ok);
    tensors_.push_back(t);
    return t;
  }

  void upload(Tensor* t, const std::vector<float>& values) {
    std::vector<uint16_t> half(values.size());
    const void* src = values.data();
    if (dtype() == slim_c10::ScalarType::Half) {
      for (size_t i = 0; i < values.size(); ++i) {
        half[i] = __half_as_ushort(__float2half(values[i]));
      }
      src = half.data();
    } else if (dtype() == slim_c10::ScalarType::BFloat16) {
      for (size_t i = 0; i < values.size(); ++i) {
        half[i] = __bfloat16_as_ushort(__float2bfloat16(values[i]));
      }
      src = half.data();
    }
    ASSERT_EQ(
        cudaMemcpy(
            t->data_ptr(),
            src,
            values.size() * t->itemsize(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
  }

  // Every element of the contiguous buffer `base`, as floats.
  std::vector<float> download(Tensor* base) {
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t numel = base->numel();
    std::vector<float> out(numel);
    if (dtype() == slim_c10::ScalarType::Float) {
      EXPECT_EQ(
          cudaMemcpy(
              out.data(),
              base->data_ptr(),
              numel * sizeof(float),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      return out;
    }
    std::vector<uint16_t> half(numel);
    EXPECT_EQ(
        cudaMemcpy(
            half.data(),
            base->data_ptr(),
            numel * sizeof(uint16_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < numel; ++i) {
      out[i] = dtype() == slim_c10::ScalarType::Half
          ? __half2float(__ushort_as_half(half[i]))
          : __bfloat162float(__ushort_as_bfloat16(half[i]));
    }
    return out;
  }

  // Element (b, r, c) of a 2-D or 3-D view of a buffer() filled with the
  // pattern, recomputed from its strides.
  static float at(Tensor* t, int64_t b, int64_t r, int64_t c) {
    const int64_t d = static_cast<int64_t>(t->dim());
    const int64_t batch_stride = d == 3 ? t->stride(0) : 0;
    return pattern(
        b * batch_stride + r * t->stride(d - 2) + c * t->stride(d - 1));
  }

  // Checks out (contiguous base `out_base` viewed as `out`) against
  // beta * self_value(b, r, c) + alpha * a @ b.
  template <typename SelfValue>
  void expect_product(
      Tensor* out_base,
      Tensor* out,
      Tensor* a,
      Tensor* b,
      float alpha,
      float beta,
      SelfValue self_value) {
    const std::vector<float> got = download(out_base);
    const int64_t d = static_cast<int64_t>(out->dim());
    const int64_t batch = d == 3 ? out->size(0) : 1;
    const int64_t rows = out->size(d - 2);
    const int64_t cols = out->size(d - 1);
    const int64_t inner = a->size(d - 1);
    const int64_t out_batch_stride = d == 3 ? out->stride(0) : 0;
    for (int64_t n = 0; n < batch; ++n) {
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          float expected = 0;
          for (int64_t k = 0; k < inner; ++k) {
            expected += at(a, n, r, k) * at(b, n, k, c);
          }
          expected = alpha * expected + beta * self_value(n, r, c);
          const int64_t index = n * out_batch_stride + r * out->stride(d - 2) +
              c * out->stride(d - 1);
          ASSERT_EQ(got[index], expected)
              << "batch " << n << " row " << r << " col " << c;
        }
      }
    }
  }

  std::vector<Tensor*> tensors_;
};

constexpr int64_t M = 5;
constexpr int64_t K = 7;
constexpr int64_t N = 3;

TEST_P(AOTITorchCudaMmTest, MmContiguous) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

// A view that starts one element into its storage is not 16-byte aligned.
// Sizes are multiples of 16 so that only the pointer is misaligned.
TEST_P(AOTITorchCudaMmTest, MmMisalignedOperand) {
  constexpr int64_t S = 64;
  Tensor* a_base = buffer(S * S + 1);
  Tensor* a = nullptr;
  const std::vector<int64_t> sizes = {S, S};
  const std::vector<int64_t> strides = {S, 1};
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          a_base, 2, sizes.data(), strides.data(), 1, &a),
      Error::Ok);
  tensors_.push_back(a);
  Tensor* b = view(buffer(S * S), {S, S}, {S, 1});
  Tensor* out_base = buffer(S * S);
  Tensor* out = view(out_base, {S, S}, {S, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  const std::vector<float> got = download(out_base);
  for (int64_t r = 0; r < S; ++r) {
    for (int64_t c = 0; c < S; ++c) {
      float expected = 0;
      for (int64_t k = 0; k < S; ++k) {
        expected += pattern(1 + r * S + k) * pattern(k * S + c);
      }
      ASSERT_EQ(got[r * S + c], expected) << "row " << r << " col " << c;
    }
  }
}

// A linear layer's weight arrives transposed (column-major), and a slice keeps
// its parent's row stride.
TEST_P(AOTITorchCudaMmTest, MmTransposedAndSlicedOperands) {
  Tensor* a = view(buffer(M * K), {M, K}, {1, M});
  Tensor* b = view(buffer(K * (N + 4)), {K, N}, {N + 4, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, MmColumnMajorResult) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {1, M});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, AddmmBiasRow) {
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {1, K});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 2.0), Error::Ok);
  expect_product(out_base, out, a, b, 2, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

// A linear layer with one output still adds its bias in the epilogue.
TEST_P(AOTITorchCudaMmTest, AddmmBiasRowOneColumn) {
  Tensor* bias = view(buffer(1), {1}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K), {K, 1}, {1, K});
  Tensor* out_base = buffer(M);
  Tensor* out = view(out_base, {M, 1}, {1, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 2.0), Error::Ok);
  expect_product(out_base, out, a, b, 2, 1, [](auto...) { return pattern(0); });
}

TEST_P(AOTITorchCudaMmTest, AddmmExpandedBias) {
  Tensor* bias = view(buffer(N), {M, N}, {0, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmFullSelf) {
  Tensor* self = view(buffer(M * N), {M, N}, {N, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.5, 2.0), Error::Ok);
  expect_product(
      out_base, out, a, b, 2, 0.5, [](int64_t, int64_t r, int64_t c) {
        return pattern(r * N + c);
      });
}

TEST_P(AOTITorchCudaMmTest, AddmmScaledBiasRow) {
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 0.5, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0.5, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

// A bias needs a row-major result for the epilogue, so this one is copied.
TEST_P(AOTITorchCudaMmTest, AddmmBiasRowColumnMajorResult) {
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {1, M});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

// A view that starts one element into its storage is not 16-byte aligned.
TEST_P(AOTITorchCudaMmTest, AddmmMisalignedBiasRow) {
  Tensor* bias_base = buffer(N + 1);
  Tensor* bias = nullptr;
  const int64_t size = N;
  const int64_t stride = 1;
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(bias_base, 1, &size, &stride, 1, &bias),
      Error::Ok);
  tensors_.push_back(bias);
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(1 + c);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmColumnBias) {
  Tensor* bias = view(buffer(M), {M, 1}, {1, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t r, int64_t) {
    return pattern(r);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmZeroDimSelf) {
  Tensor* self = view(buffer(1), {}, {});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 2.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 2, [](auto...) { return pattern(0); });
}

TEST_P(AOTITorchCudaMmTest, AddmmScalarSelf) {
  Tensor* self = view(buffer(1), {1, 1}, {1, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 2.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 2, [](auto...) { return pattern(0); });
}

// self laid out column-major while out is row-major.
TEST_P(AOTITorchCudaMmTest, AddmmTransposedSelf) {
  Tensor* self = view(buffer(M * N), {M, N}, {1, M});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.5, 2.0), Error::Ok);
  expect_product(
      out_base, out, a, b, 2, 0.5, [](int64_t, int64_t r, int64_t c) {
        return pattern(r + c * M);
      });
}

// With one row a full self is also a bias row, but beta is not 1.
TEST_P(AOTITorchCudaMmTest, AddmmOneRowScaledSelf) {
  Tensor* self = view(buffer(N), {1, N}, {N, 1});
  Tensor* a = view(buffer(K), {1, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {1, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.5, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0.5, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

// With one row, self == out is a bias row, but the epilogue may read the bias
// after the matmul has written out. The split-K algorithms cuBLASLt picks for
// this shape on an H100 do, so it must take the in-place path instead.
TEST_P(AOTITorchCudaMmTest, AddmmInPlaceOneRow) {
  if (dtype() != slim_c10::ScalarType::Float) {
    GTEST_SKIP() << "the sums are exact in float32 only";
  }
  constexpr int64_t Cols = 256;
  constexpr int64_t Inner = 16384;
  Tensor* a = view(buffer(Inner), {1, Inner}, {Inner, 1});
  Tensor* b = view(buffer(Inner * Cols), {Inner, Cols}, {1, Inner});
  Tensor* out_base = buffer(Cols);
  Tensor* out = view(out_base, {1, Cols}, {Cols, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, out, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmInPlace) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, out, a, b, 0.5, 1.0), Error::Ok);
  expect_product(
      out_base, out, a, b, 1, 0.5, [](int64_t, int64_t r, int64_t c) {
        return pattern(r * N + c);
      });
}

// With beta == 0 PyTorch ignores self, so a NaN in it, or in out before the
// call, must not reach the result.
TEST_P(AOTITorchCudaMmTest, AddmmZeroBetaIgnoresSelf) {
  Tensor* self = view(buffer(M * N), {M, N}, {N, 1});
  upload(self, std::vector<float>(M * N, NAN));
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  upload(out_base, std::vector<float>(M * N, NAN));
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.0, 2.0), Error::Ok);
  expect_product(out_base, out, a, b, 2, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, RejectsSelfThatDoesNotBroadcast) {
  Tensor* self = view(buffer(N + 1), {N + 1}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out = view(buffer(M * N), {M, N}, {N, 1});
  EXPECT_EQ(
      aoti_torch_cuda_addmm_out(out, self, a, b, 1.0, 1.0),
      Error::InvalidArgument);
}

// Slices of a wider matrix, as x[:, :0] @ w[:0, :] gives. The product of an
// empty inner dimension is zeros, and addmm scales self by beta.
TEST_P(AOTITorchCudaMmTest, MmEmptyInnerDimension) {
  Tensor* a = view(buffer(M * K), {M, 0}, {K, 1});
  Tensor* b = view(buffer(K * N), {0, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, AddmmEmptyInnerDimension) {
  Tensor* self = view(buffer(M * N), {M, N}, {N, 1});
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(M * K), {M, 0}, {K, 1});
  Tensor* b = view(buffer(K * N), {0, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.5, 1.0), Error::Ok);
  expect_product(
      out_base, out, a, b, 1, 0.5, [](int64_t, int64_t r, int64_t c) {
        return pattern(r * N + c);
      });
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

TEST_P(AOTITorchCudaMmTest, BmmEmptyInnerDimension) {
  constexpr int64_t B = 3;
  Tensor* a = view(buffer(B * M * K), {B, M, 0}, {M * K, K, 1});
  Tensor* b = view(buffer(B * K * N), {B, 0, N}, {K * N, N, 1});
  Tensor* out_base = buffer(B * M * N);
  Tensor* out = view(out_base, {B, M, N}, {M * N, N, 1});
  ASSERT_EQ(aoti_torch_cuda_bmm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

// 1 + 2^-12 rounds to 1 in TF32, so a TF32 product gives 64 where full
// precision gives 64 * (1 + 2^-11) = 64.03125.
TEST_P(AOTITorchCudaMmTest, MmFloat32IsFullPrecision) {
  if (dtype() != slim_c10::ScalarType::Float) {
    GTEST_SKIP() << "TF32 only applies to float32";
  }
  constexpr int64_t S = 64;
  const float value = 1.0f + 1.0f / 4096;
  Tensor* a = view(buffer(S * S), {S, S}, {S, 1});
  Tensor* b = view(buffer(S * S), {S, S}, {S, 1});
  upload(a, std::vector<float>(S * S, value));
  upload(b, std::vector<float>(S * S, value));
  Tensor* out_base = buffer(S * S);
  Tensor* out = view(out_base, {S, S}, {S, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  for (float got : download(out_base)) {
    ASSERT_NEAR(got, 64.03125f, 1e-4f);
  }
}

TEST_P(AOTITorchCudaMmTest, Bmm) {
  constexpr int64_t B = 3;
  Tensor* a = view(buffer(B * M * K), {B, M, K}, {M * K, K, 1});
  Tensor* b = view(buffer(B * K * N), {B, K, N}, {K * N, 1, K});
  Tensor* out_base = buffer(B * M * N);
  Tensor* out = view(out_base, {B, M, N}, {M * N, N, 1});
  ASSERT_EQ(aoti_torch_cuda_bmm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, BmmBroadcastOperand) {
  constexpr int64_t B = 3;
  Tensor* a = view(buffer(B * M * K), {B, M, K}, {M * K, K, 1});
  Tensor* b = view(buffer(K * N), {B, K, N}, {0, N, 1});
  Tensor* out_base = buffer(B * M * N);
  Tensor* out = view(out_base, {B, M, N}, {M * N, N, 1});
  ASSERT_EQ(aoti_torch_cuda_bmm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

// The workspace comes from the stream-ordered allocator, so a call can be
// recorded into a CUDA graph and replayed, as the backend does. A long inner
// dimension is what makes cuBLASLt ask for a workspace. mat2 is the first S
// columns of an identity, so every partial sum is exact in every dtype.
TEST_P(AOTITorchCudaMmTest, MmCapturesIntoCudaGraph) {
  constexpr int64_t S = 64;
  constexpr int64_t L = 8192;
  Tensor* a = view(buffer(S * L), {S, L}, {L, 1});
  Tensor* b = view(buffer(L * S), {L, S}, {1, L});
  std::vector<float> identity(L * S, 0.0f);
  for (int64_t c = 0; c < S; ++c) {
    identity[c * L + c] = 1.0f;
  }
  upload(b, identity);
  Tensor* out_base = buffer(S * S);
  aoti_torch_zero_(out_base);
  Tensor* out = view(out_base, {S, S}, {S, 1});

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(setCurrentCUDAStream(stream, 0), Error::Ok);
  ASSERT_EQ(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  const Error captured = aoti_torch_cuda_mm_out(out, a, b);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  ASSERT_EQ(captured, Error::Ok);
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(
      cudaGraphInstantiate(
          &exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch),
      cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> got = download(out_base);
  for (int64_t r = 0; r < S; ++r) {
    for (int64_t c = 0; c < S; ++c) {
      ASSERT_EQ(got[r * S + c], pattern(r * L + c))
          << "row " << r << " col " << c;
    }
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

TEST_P(AOTITorchCudaMmTest, RejectsMismatchedShapes) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer((K + 1) * N), {K + 1, N}, {N, 1});
  Tensor* out = view(buffer(M * N), {M, N}, {N, 1});
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::InvalidArgument);
}

TEST_P(AOTITorchCudaMmTest, RejectsOperandThatIsNotAMatrixLayout) {
  // Neither dimension is unit-stride.
  Tensor* a = view(buffer(2 * M * K), {M, K}, {2 * K, 2});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out = view(buffer(M * N), {M, N}, {N, 1});
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::NotSupported);
}

INSTANTIATE_TEST_SUITE_P(
    Dtypes,
    AOTITorchCudaMmTest,
    ::testing::Values(
        slim_c10::ScalarType::Float,
        slim_c10::ScalarType::Half,
        slim_c10::ScalarType::BFloat16));

} // namespace
