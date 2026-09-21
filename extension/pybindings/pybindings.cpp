/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/devtools/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/module/bundled_module.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/pybindings/dlpack.h>
#include <executorch/extension/pybindings/pybindings_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

// A build defining EXECUTORCH_PYBINDINGS_WITHOUT_TORCH links no torch. It still
// takes and returns torch tensors, reading them through torch's own Python API
// rather than in C++. ATen mode is the runtime built on torch's tensor type, so
// the two cannot be combined.
#if defined(USE_ATEN_LIB) && defined(EXECUTORCH_PYBINDINGS_WITHOUT_TORCH)
#error "USE_ATEN_LIB builds on torch and cannot be built without it"
#endif

#ifndef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/core/functional.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#endif // !EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

#if !defined(USE_ATEN_LIB) && !defined(EXECUTORCH_PYBINDINGS_WITHOUT_TORCH)
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <executorch/extension/aten_util/aten_bridge.h>
#endif

/// Throws a runtime_error with the provided message if `error` is not `Ok`.
#define THROW_IF_ERROR(error, message, ...)                       \
  ({                                                              \
    if ((error) != Error::Ok) {                                   \
      char msg_buf[128];                                          \
      snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
      /* pybind will convert this to a python exception. */       \
      throw std::runtime_error(msg_buf);                          \
    }                                                             \
  })

#define THROW_INDEX_IF_ERROR(error, message, ...)                 \
  ({                                                              \
    if ((error) != Error::Ok) {                                   \
      char msg_buf[128];                                          \
      snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
      /* pybind will convert this to a python exception. */       \
      throw std::out_of_range(msg_buf);                           \
    }                                                             \
  })

namespace py = pybind11;
using ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_class;
using ::executorch::ET_RUNTIME_NAMESPACE::get_backend_name;
using ::executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends;
using ::executorch::ET_RUNTIME_NAMESPACE::get_registered_kernels;
using ::executorch::ET_RUNTIME_NAMESPACE::Kernel;
using ::executorch::ET_RUNTIME_NAMESPACE::Method;
using ::executorch::ET_RUNTIME_NAMESPACE::MethodMeta;
using ::executorch::ET_RUNTIME_NAMESPACE::Program;
using ::executorch::ET_RUNTIME_NAMESPACE::TensorInfo;
using ::executorch::extension::BufferDataLoader;
using ::executorch::extension::MallocMemoryAllocator;
using ::executorch::extension::MmapDataLoader;
using ::executorch::extension::ET_BUNDLED_MODULE_NAMESPACE::BundledModule;
using ::executorch::extension::pybindings::PyDataLoader;
using ::executorch::runtime::DataLoader;
using ::executorch::runtime::DeviceMemoryBuffer;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::EventTracerDebugLogLevel;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::prof_result_t;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using ::executorch::runtime::Tag;
using torch::executor::etdump_result;
using torch::executor::ETDumpGen;

#if !defined(USE_ATEN_LIB) && !defined(EXECUTORCH_PYBINDINGS_WITHOUT_TORCH)
using ::executorch::extension::alias_attensor_to_etensor;
using ::executorch::extension::alias_etensor_to_attensor;
using ::executorch::extension::torch_to_executorch_device;
using ::executorch::extension::torch_to_executorch_scalar_type;
#endif // !USE_ATEN_LIB && !EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

namespace executorch {
namespace extension {
namespace pybindings {

namespace {

#ifndef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
// A negated or conjugated view does not hold the values it reports: torch
// applies the sign or the conjugate when reading the memory, and nothing below
// does, so the method would run on the underlying values instead.
void reject_lazy_view(
    const at::Tensor& tensor,
    size_t input_index,
    const char* method_name) {
  if (tensor.is_neg() || tensor.is_conj()) {
    throw std::runtime_error(
        "Input " + std::to_string(input_index) + " for method " + method_name +
        " is a negated or conjugated view, whose memory holds different values than the tensor does. Call .resolve_neg() or .resolve_conj() on it first.");
  }
}

// A tensor can describe elements it has no memory for, and the conversions
// below assert rather than raise when they meet one, which ends the process. A
// subclass made with _make_wrapper_subclass describes a tensor without
// allocating one, which is the way to reach that.
void reject_tensor_without_data(
    const at::Tensor& tensor,
    size_t input_index,
    const char* method_name) {
  // Asked without taking a mutable pointer, which would copy the memory of a
  // tensor that shares it.
  if (tensor.numel() != 0 && tensor.const_data_ptr() == nullptr) {
    throw std::runtime_error(
        "Input " + std::to_string(input_index) + " for method " + method_name +
        " has no data, so there is nothing for the method to read.");
  }
}

void* mutable_tensor_data_ptr_no_cow(at::Tensor& tensor) {
  if (tensor.numel() == 0) {
    return nullptr;
  }

  auto* storage_data = static_cast<char*>(tensor.unsafeGetTensorImpl()
                                              ->unsafe_storage()
                                              .unsafeGetStorageImpl()
                                              ->_mutable_data_ptr_no_checks()
                                              .mutable_get());
  ET_CHECK_MSG(
      storage_data != nullptr,
      "Tensor has a non-zero number of elements, but its data is not allocated");
  return storage_data + tensor.storage_offset() * tensor.itemsize();
}
#endif // !EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

// An input that is not a torch.Tensor can still hand its memory to the runtime
// through CPython's buffer protocol, which numpy arrays, `bytes`, `memoryview`
// and `array.array` all implement. This is the path that does not need torch.

// PEP 3118 spells the element type as a struct-module format code. The width of
// 'l' and 'n' is platform dependent, so the code decides signedness and
// itemsize decides the width.
std::optional<executorch::aten::ScalarType> buffer_scalar_type(
    const py::buffer_info& info) {
  const std::string& format = info.format;
  // A prefix states alignment or byte order. Native alignment is what a bare
  // code already means, and the host's own byte order is the only one the
  // runtime can read, so anything else has to be refused rather than
  // reinterpreted. `ctypes` does spell the native order out.
  const bool little_endian = []() {
    const uint16_t probe = 1;
    return *reinterpret_cast<const uint8_t*>(&probe) == 1;
  }();
  size_t code_index = 0;
  if (!format.empty()) {
    const char prefix = format[0];
    if (prefix == '@' || prefix == '=' || (prefix == '<' && little_endian) ||
        (prefix == '>' && !little_endian)) {
      code_index = 1;
    } else if (prefix == '<' || prefix == '>' || prefix == '!') {
      return std::nullopt;
    }
  }
  if (format.size() != code_index + 1 && format[code_index] != 'Z') {
    return std::nullopt;
  }
  switch (format[code_index]) {
    case '?':
      return executorch::aten::ScalarType::Bool;
    case 'b':
    case 'h':
    case 'i':
    case 'l':
    case 'q':
    case 'n':
      switch (info.itemsize) {
        case 1:
          return executorch::aten::ScalarType::Char;
        case 2:
          return executorch::aten::ScalarType::Short;
        case 4:
          return executorch::aten::ScalarType::Int;
        case 8:
          return executorch::aten::ScalarType::Long;
      }
      return std::nullopt;
    case 'B':
    case 'H':
    case 'I':
    case 'L':
    case 'Q':
    case 'N':
      switch (info.itemsize) {
        case 1:
          return executorch::aten::ScalarType::Byte;
        case 2:
          return executorch::aten::ScalarType::UInt16;
        case 4:
          return executorch::aten::ScalarType::UInt32;
        case 8:
          return executorch::aten::ScalarType::UInt64;
      }
      return std::nullopt;
    case 'e':
    case 'f':
    case 'd':
      switch (info.itemsize) {
        case 2:
          return executorch::aten::ScalarType::Half;
        case 4:
          return executorch::aten::ScalarType::Float;
        case 8:
          return executorch::aten::ScalarType::Double;
      }
      return std::nullopt;
    case 'Z':
      // 'Z' prefixes the component code, so the pair is two characters.
      if (format.size() != code_index + 2) {
        return std::nullopt;
      }
      switch (info.itemsize) {
        case 8:
          return executorch::aten::ScalarType::ComplexFloat;
        case 16:
          return executorch::aten::ScalarType::ComplexDouble;
      }
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

// The format code a reader needs to interpret the bytes of a scalar type, or
// nothing when the buffer protocol has no code for it, as with bfloat16. Each
// caller decides what to do then, and none of them refuses the dtype.
std::optional<const char*> buffer_format(
    executorch::aten::ScalarType scalar_type) {
  switch (scalar_type) {
    case executorch::aten::ScalarType::Bool:
      return "?";
    case executorch::aten::ScalarType::Char:
      return "b";
    case executorch::aten::ScalarType::Byte:
      return "B";
    case executorch::aten::ScalarType::Short:
      return "h";
    case executorch::aten::ScalarType::UInt16:
      return "H";
    case executorch::aten::ScalarType::Int:
      return "i";
    case executorch::aten::ScalarType::UInt32:
      return "I";
    case executorch::aten::ScalarType::Long:
      return "q";
    case executorch::aten::ScalarType::UInt64:
      return "Q";
    case executorch::aten::ScalarType::Half:
      return "e";
    case executorch::aten::ScalarType::Float:
      return "f";
    case executorch::aten::ScalarType::Double:
      return "d";
    case executorch::aten::ScalarType::ComplexFloat:
      return "Zf";
    case executorch::aten::ScalarType::ComplexDouble:
      return "Zd";
    default:
      return std::nullopt;
  }
}

// Whether `strides`, in bytes, are the ones `dim_order` implies for `shape`.
// Dimensions of size one carry no layout information, so their strides are not
// compared.
bool strides_have_dim_order(
    const std::vector<py::ssize_t>& shape,
    const std::vector<py::ssize_t>& strides,
    py::ssize_t itemsize,
    const std::vector<executorch::aten::DimOrderType>& dim_order) {
  py::ssize_t stride = itemsize;
  for (auto dim = dim_order.rbegin(); dim != dim_order.rend(); ++dim) {
    if (shape[*dim] > 1 && strides[*dim] != stride) {
      return false;
    }
    stride *= shape[*dim];
  }
  return true;
}

/// Refuses an input whose memory is not laid out the way the method expects.
///
/// Nothing further down compares layouts: the runtime either copies an input by
/// byte count or hands its pointer over, so an input laid out differently from
/// the one the method was exported with is read in the wrong order and gives
/// wrong numbers with no error. It is refused here because this is the last
/// place that holds the source's own layout, and the caller who built the
/// source is the only one who can lay it out differently.
void reject_unexpected_layout(
    const std::vector<py::ssize_t>& shape,
    const std::vector<py::ssize_t>& strides,
    py::ssize_t itemsize,
    size_t input_index,
    const std::string& method_name,
    const Result<TensorInfo>& expected) {
  if (!expected.ok()) {
    return;
  }
  const std::string prefix =
      "Input " + std::to_string(input_index) + " for method " + method_name;
  const auto dim_order = expected->dim_order();
  if (dim_order.size() != shape.size()) {
    throw std::runtime_error(
        prefix + " has " + std::to_string(shape.size()) +
        " dimensions, but the method expects " +
        std::to_string(dim_order.size()) + ".");
  }
  // A tensor with no elements has nothing laid out, and what a source reports
  // for its strides carries no meaning: torch says (1, 1) for shape (2, 0)
  // while numpy says (0, 0).
  if (std::any_of(shape.begin(), shape.end(), [](py::ssize_t size) {
        return size == 0;
      })) {
    return;
  }
  // Compared as strides and never as dim orders, because a dim order cannot
  // express the exemption below: a size-one dimension may legally sit anywhere
  // in an order, so two orders that disagree only about where they put it still
  // describe the same bytes.
  py::ssize_t stride = itemsize;
  for (size_t i = dim_order.size(); i-- > 0;) {
    const size_t dim = dim_order[i];
    // A size-one dimension is never stepped along, so its stride is not part of
    // where the values are.
    if (shape[dim] > 1 && strides[dim] != stride) {
      throw std::runtime_error(
          prefix + " has stride " + std::to_string(strides[dim]) +
          " bytes at dimension " + std::to_string(dim) +
          ", but the method expects " + std::to_string(stride) +
          ". Pass it in the memory layout the method was exported with.");
    }
    stride *= shape[dim];
  }
}

#ifndef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
/// The same refusal for a build that links torch, whose input path holds the
/// tensor itself rather than a description of its memory.
void reject_unexpected_layout(
    const at::Tensor& tensor,
    size_t input_index,
    const std::string& method_name,
    const Result<TensorInfo>& expected) {
  const auto itemsize = static_cast<py::ssize_t>(tensor.element_size());
  // torch counts strides in elements, the check above in bytes.
  std::vector<py::ssize_t> strides;
  strides.reserve(tensor.strides().size());
  for (const auto stride : tensor.strides()) {
    strides.push_back(static_cast<py::ssize_t>(stride) * itemsize);
  }
  reject_unexpected_layout(
      std::vector<py::ssize_t>(tensor.sizes().begin(), tensor.sizes().end()),
      strides,
      itemsize,
      input_index,
      method_name,
      expected);
}
#endif // !EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

/// Refuses a shape the runtime cannot describe, before any of it is narrowed.
///
/// The runtime keeps sizes in a 32 bit signed type. A larger dimension wraps
/// when it narrows, and a wrapped size reads as negative, which fails an
/// assertion deep in the runtime and ends the process instead of raising. A
/// tensor with no elements can carry such a dimension without using any memory,
/// so this is reachable without allocating anything.
template <typename Sizes>
void reject_unrepresentable_shape(
    const Sizes& shape,
    size_t index,
    const std::string& method_name) {
  constexpr auto kMaxSize =
      std::numeric_limits<executorch::aten::SizesType>::max();
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    const auto size = static_cast<int64_t>(shape[dim]);
    if (size < 0 || size > static_cast<int64_t>(kMaxSize)) {
      throw std::runtime_error(
          "Input " + std::to_string(index) + " for method " + method_name +
          " has size " + std::to_string(size) + " at dimension " +
          std::to_string(dim) + ", which is outside what the runtime can" +
          " describe. The largest size it holds is " +
          std::to_string(static_cast<int64_t>(kMaxSize)) + ".");
    }
  }
}

// The layouts the runtime can describe: the default order, and the
// channels-last order a method of four or five dimensions may be exported with.
std::optional<std::vector<executorch::aten::DimOrderType>>
dim_order_from_strides(
    const std::vector<py::ssize_t>& shape,
    const std::vector<py::ssize_t>& strides,
    py::ssize_t itemsize) {
  // Past this many dimensions the runtime cannot compute strides at all, and
  // asking it to would abort the process rather than raise.
  if (shape.size() > executorch::runtime::kTensorDimensionLimit) {
    return std::nullopt;
  }
  std::vector<executorch::aten::DimOrderType> dim_order(shape.size());
  std::iota(dim_order.begin(), dim_order.end(), 0);
  // A tensor with no elements has nothing laid out, and the strides reported
  // for one carry no meaning: torch says (1, 1) for shape (2, 0) while numpy
  // says (0, 0).
  for (const auto size : shape) {
    if (size == 0) {
      return dim_order;
    }
  }
  if (strides_have_dim_order(shape, strides, itemsize, dim_order)) {
    return dim_order;
  }
  // The runtime reads a channels-last order at four and five dimensions and at
  // no other rank, so no other rank has one to compare against.
  if (shape.size() == 4) {
    dim_order = {0, 2, 3, 1};
  } else if (shape.size() == 5) {
    dim_order = {0, 2, 3, 4, 1};
  } else {
    return std::nullopt;
  }
  if (strides_have_dim_order(shape, strides, itemsize, dim_order)) {
    return dim_order;
  }
  return std::nullopt;
}

#if !defined(USE_ATEN_LIB) && !defined(EXECUTORCH_PYBINDINGS_WITHOUT_TORCH)
// The runtime layout of a torch tensor's memory. Decided from the strides
// rather than from torch's memory formats, so that a tensor and a buffer
// holding the same bytes are described the same way, and so that the ranks a
// channels-last order exists for are stated in one place.
std::optional<std::vector<executorch::aten::DimOrderType>> tensor_dim_order(
    const at::Tensor& tensor) {
  const auto itemsize = static_cast<py::ssize_t>(tensor.element_size());
  // torch counts strides in elements, the layouts above in bytes.
  std::vector<py::ssize_t> strides;
  strides.reserve(tensor.strides().size());
  for (const auto stride : tensor.strides()) {
    strides.push_back(static_cast<py::ssize_t>(stride) * itemsize);
  }
  return dim_order_from_strides(
      std::vector<py::ssize_t>(tensor.sizes().begin(), tensor.sizes().end()),
      strides,
      itemsize);
}
#endif // !USE_ATEN_LIB && !EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

// Describes the buffer's memory as a tensor, or an owned copy of it when the
// caller asks for one. The buffer protocol pins that memory for as long as
// `info` lives, so pointing at it is safe while the view is held.
//
// A flat buffer of bytes whose size matches the input is taken as the input's
// raw bytes, and its dtype, sizes and layout come from `expected`. That is the
// way to pass a dtype the buffer protocol cannot spell, bfloat16 among them.
TensorPtr tensor_from_buffer(
    const py::buffer_info& info,
    size_t input_index,
    const char* method_name,
    const Result<TensorInfo>& expected,
    bool copy) {
  reject_unrepresentable_shape(info.shape, input_index, method_name);
  const auto nbytes = static_cast<size_t>(info.size * info.itemsize);
  // Raw bytes are accepted only for a dtype no format code can name, which is
  // the one case where nothing else can express the input. Allowing it whenever
  // the byte count happened to match would silently reinterpret, say, a uint8
  // image as float32.
  const bool raw_bytes = expected.ok() &&
      !buffer_format(expected->scalar_type()).has_value() && info.ndim == 1 &&
      info.itemsize == 1 &&
      info.format == py::format_descriptor<uint8_t>::format() &&
      // A strided byte view holds its bytes apart, so reading it as if they
      // were packed takes the gaps as data.
      info.strides[0] == 1 && nbytes == expected->nbytes();

  std::vector<executorch::aten::SizesType> sizes;
  std::vector<executorch::aten::DimOrderType> dim_order;
  executorch::aten::ScalarType scalar_type;
  if (raw_bytes) {
    // The shape comes from the method here rather than from the buffer, so it
    // still has to be one the runtime can compute strides for. Past its limit
    // the computation fails an assertion, which takes the process down.
    if (expected->sizes().size() > executorch::runtime::kTensorDimensionLimit) {
      throw std::runtime_error(
          "Input " + std::to_string(input_index) + " for method " +
          method_name + " has " + std::to_string(expected->sizes().size()) +
          " dimensions, which is more than the runtime can describe.");
    }
    sizes.assign(expected->sizes().begin(), expected->sizes().end());
    dim_order.assign(
        expected->dim_order().begin(), expected->dim_order().end());
    scalar_type = expected->scalar_type();
  } else {
    const auto buffer_type = buffer_scalar_type(info);
    if (!buffer_type.has_value()) {
      std::string message = "Input " + std::to_string(input_index) +
          " for method " + method_name + " has buffer format '" + info.format +
          "' of " + std::to_string(info.itemsize) +
          " bytes, which no dtype describes.";
      // Raw bytes are only a way in when the dtype the method expects has no
      // format code either, so offering them otherwise sends the caller into a
      // second refusal.
      if (expected.ok() &&
          !buffer_format(expected->scalar_type()).has_value()) {
        message +=
            " No buffer format names the dtype this input takes, so pass "
            "a flat buffer of " +
            std::to_string(expected->nbytes()) + " raw bytes.";
      } else {
        message +=
            " Pass a buffer whose format code names the dtype the method expects, in the host's own byte order.";
      }
      throw std::runtime_error(message);
    }
    auto buffer_order =
        dim_order_from_strides(info.shape, info.strides, info.itemsize);
    if (!buffer_order.has_value()) {
      throw std::runtime_error(
          "Input " + std::to_string(input_index) + " for method " +
          method_name + " has " + std::to_string(info.ndim) +
          " dimensions and strides no runtime layout describes. Pass it in the default order or, at four or five dimensions, in channels-last order.");
    }
    sizes.assign(info.shape.begin(), info.shape.end());
    dim_order = std::move(*buffer_order);
    scalar_type = *buffer_type;
    // The runtime refuses a dtype the method did not ask for, but only with an
    // error code, and passing bytes for a float input is a common enough
    // mistake to name here.
    if (expected.ok() && expected->scalar_type() != scalar_type) {
      std::string message = "Input " + std::to_string(input_index) +
          " for method " + method_name + " is " +
          executorch::runtime::toString(scalar_type) +
          ", but the method expects " +
          executorch::runtime::toString(expected->scalar_type()) + ".";
      if (buffer_format(expected->scalar_type()).has_value()) {
        message += " Pass a buffer of that dtype.";
      } else {
        // No format code names that dtype, so raw bytes are the only way in,
        // and they have to fill the input exactly. A dynamically shaped input
        // reports the largest shape it takes, which is the size that has to be
        // passed.
        message +=
            " No buffer format names that dtype, so pass a flat buffer of " +
            std::to_string(expected->nbytes()) +
            " raw bytes, which is what this input takes at its largest shape.";
      }
      throw std::runtime_error(message);
    }
    reject_unexpected_layout(
        info.shape,
        info.strides,
        info.itemsize,
        input_index,
        method_name,
        expected);
  }

  if (copy) {
    const auto* bytes = static_cast<const uint8_t*>(info.ptr);
    return make_tensor_ptr(
        std::move(sizes),
        std::vector<uint8_t>(bytes, bytes + nbytes),
        std::move(dim_order),
        {},
        scalar_type,
        executorch::aten::TensorShapeDynamism::STATIC);
  }
  return for_blob(info.ptr, std::move(sizes), scalar_type)
      .dim_order(std::move(dim_order))
      .dynamism(executorch::aten::TensorShapeDynamism::STATIC)
      .make_tensor_ptr();
}

// Whether an object is a torch.Tensor, an instance of a subclass included.
//
// Asked of the type rather than of the name the type prints, because dispatch
// has to agree with the converter that follows it, which accepts any instance.
// A subclass whose name does not match is taken for something else, and since
// every torch tensor also offers DLPack it lands on that path instead, which
// hands over the memory without the checks a tensor needs. Torch is not
// imported here, only recognised when the caller has already imported it, since
// an instance cannot exist before that.
bool is_torch_tensor(const py::handle& object) {
  const py::object modules = py::module_::import("sys").attr("modules");
  if (!py::cast<bool>(modules.attr("__contains__")("torch"))) {
    return false;
  }
  return py::isinstance(object, modules["torch"].attr("Tensor"));
}

// One input, or a sequence of them. A buffer counts as one input even though a
// numpy array also satisfies the sequence protocol, and anything that is not a
// sequence at all is one input too, which is what lets a single tensor or a
// single output be passed straight back in.
py::sequence as_input_sequence(const py::object& inputs) {
  if (!py::isinstance<py::buffer>(inputs) &&
      py::isinstance<py::sequence>(inputs)) {
    return py::cast<py::sequence>(inputs);
  }
  py::list single;
  single.append(inputs);
  return single;
}

/// Whether this input has to be copied rather than pointed at.
///
/// A method with planned input memory copies the input itself, so the choice
/// only matters for one exported without it, where the runtime keeps the
/// pointer until it runs. `pins_memory` says whether the source keeps its
/// memory in place and lets the runtime write to it. A writable buffer view
/// does both, so it is pointed at. A tensor object does neither, since it can
/// be resized, re-pointed, or have its storage freed while the object lives,
/// and a read only buffer refuses the writing, so both of those are copied.
bool input_needs_owned_copy(
    const Result<TensorInfo>& expected,
    bool pins_memory) {
  if (!expected.ok()) {
    return true;
  }
  if (expected->is_memory_planned()) {
    return false;
  }
  return !pins_memory;
}

/// A DLPack capsule taken from a producer, and the promise to give it back.
///
/// The protocol says a consumer claims a capsule by renaming it, which tells
/// the producer not to free what is inside, and then calls the deleter when it
/// is finished. Holding that in one object means the promise is kept even when
/// a conversion throws.
class DLPackInput final {
 public:
  explicit DLPackInput(const py::object& source) {
    py::object capsule = source.attr("__dlpack__")();
    auto* raw = PyCapsule_GetPointer(capsule.ptr(), "dltensor");
    if (raw == nullptr) {
      PyErr_Clear();
      throw std::runtime_error(
          "This object offers __dlpack__ but did not hand over memory that can be read. A capsule can only be taken once.");
    }
    // Claimed: the producer must not free it, and this object must.
    PyCapsule_SetName(capsule.ptr(), "used_dltensor");
    managed_ = static_cast<DLManagedTensor*>(raw);
  }

  DLPackInput(DLPackInput&& other) noexcept : managed_(other.managed_) {
    other.managed_ = nullptr;
  }
  DLPackInput& operator=(DLPackInput&& other) noexcept {
    std::swap(managed_, other.managed_);
    return *this;
  }
  DLPackInput(const DLPackInput&) = delete;
  DLPackInput& operator=(const DLPackInput&) = delete;

  ~DLPackInput() {
    if (managed_ != nullptr && managed_->deleter != nullptr) {
      managed_->deleter(managed_);
    }
  }

  const DLTensor& tensor() const {
    return managed_->dl_tensor;
  }

 private:
  DLManagedTensor* managed_ = nullptr;
};

/// The runtime dtype a DLPack description names, if the runtime has one.
std::optional<executorch::aten::ScalarType> dlpack_scalar_type(
    const DLDataType& dtype) {
  if (dtype.lanes != 1) {
    return std::nullopt; // a vector type, which no runtime tensor describes
  }
  switch (dtype.code) {
    case kDLBool:
      return dtype.bits == 8 ? std::optional(executorch::aten::ScalarType::Bool)
                             : std::nullopt;
    case kDLInt:
      switch (dtype.bits) {
        case 8:
          return executorch::aten::ScalarType::Char;
        case 16:
          return executorch::aten::ScalarType::Short;
        case 32:
          return executorch::aten::ScalarType::Int;
        case 64:
          return executorch::aten::ScalarType::Long;
      }
      return std::nullopt;
    case kDLUInt:
      switch (dtype.bits) {
        case 8:
          return executorch::aten::ScalarType::Byte;
        case 16:
          return executorch::aten::ScalarType::UInt16;
        case 32:
          return executorch::aten::ScalarType::UInt32;
        case 64:
          return executorch::aten::ScalarType::UInt64;
      }
      return std::nullopt;
    case kDLFloat:
      switch (dtype.bits) {
        case 16:
          return executorch::aten::ScalarType::Half;
        case 32:
          return executorch::aten::ScalarType::Float;
        case 64:
          return executorch::aten::ScalarType::Double;
      }
      return std::nullopt;
    case kDLBfloat:
      return dtype.bits == 16
          ? std::optional(executorch::aten::ScalarType::BFloat16)
          : std::nullopt;
    case kDLComplex:
      switch (dtype.bits) {
        case 64:
          return executorch::aten::ScalarType::ComplexFloat;
        case 128:
          return executorch::aten::ScalarType::ComplexDouble;
      }
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

/// Describes memory a DLPack producer handed over, for the runtime.
///
/// Unlike a bare tensor object, a claimed capsule is a promise that the memory
/// stays where it is until the deleter is called, so this never needs a copy.
/// The memory may also be somewhere the host cannot read, which is the reason
/// this path exists at all.
TensorPtr tensor_from_dlpack(
    const DLPackInput& input,
    size_t input_index,
    const std::string& method_name,
    const Result<TensorInfo>& expected) {
  const std::string prefix =
      "Input " + std::to_string(input_index) + " for method " + method_name;
  const DLTensor& described = input.tensor();

  const auto scalar_type = dlpack_scalar_type(described.dtype);
  if (!scalar_type.has_value()) {
    throw std::runtime_error(
        prefix + " has a dtype of " + std::to_string(described.dtype.bits) +
        " bits that no runtime dtype describes.");
  }
  if (expected.ok() && expected->scalar_type() != *scalar_type) {
    throw std::runtime_error(
        prefix + " is " + executorch::runtime::toString(*scalar_type) +
        ", but the method expects " +
        executorch::runtime::toString(expected->scalar_type()) + ".");
  }

  executorch::aten::Device device(executorch::aten::DeviceType::CPU, 0);
  if (described.device.device_type == kDLCUDA) {
    device = executorch::aten::Device(
        executorch::aten::DeviceType::CUDA,
        static_cast<executorch::runtime::etensor::DeviceIndex>(
            described.device.device_id));
  } else if (described.device.device_type != kDLCPU) {
    throw std::runtime_error(
        prefix + " is on DLPack device kind " +
        std::to_string(described.device.device_type) +
        ", and only host and CUDA memory can be passed to a method.");
  }

  // A capsule hands its rank and its shape over as plain numbers, and reading
  // the shape below trusts both. Every other input path asks an object that
  // cannot misdescribe its own memory.
  if (described.ndim < 0) {
    throw std::runtime_error(
        prefix + " describes " + std::to_string(described.ndim) +
        " dimensions, which is not a number of dimensions.");
  }
  // Past this many dimensions the runtime cannot compute strides at all, and
  // asking it to would abort the process rather than raise.
  if (static_cast<size_t>(described.ndim) >
      executorch::runtime::kTensorDimensionLimit) {
    throw std::runtime_error(
        prefix + " has " + std::to_string(described.ndim) +
        " dimensions, which is more than the runtime can describe.");
  }
  if (described.ndim > 0 && described.shape == nullptr) {
    throw std::runtime_error(
        prefix + " describes " + std::to_string(described.ndim) +
        " dimensions but hands over no shape, so its shape cannot be read.");
  }
  const std::vector<py::ssize_t> shape(
      described.shape, described.shape + described.ndim);
  reject_unrepresentable_shape(shape, input_index, method_name);

  // DLPack counts strides in elements, and leaves them out entirely when the
  // elements are packed in the order the shape implies.
  const auto itemsize =
      static_cast<py::ssize_t>(executorch::runtime::elementSize(*scalar_type));
  std::vector<py::ssize_t> strides(described.ndim);
  if (described.strides != nullptr) {
    for (int32_t dim = 0; dim < described.ndim; ++dim) {
      strides[dim] =
          static_cast<py::ssize_t>(described.strides[dim]) * itemsize;
    }
  } else {
    py::ssize_t packed = itemsize;
    for (int32_t dim = described.ndim - 1; dim >= 0; --dim) {
      strides[dim] = packed;
      packed *= shape[dim];
    }
  }
  auto dim_order = dim_order_from_strides(shape, strides, itemsize);
  if (!dim_order.has_value()) {
    throw std::runtime_error(
        prefix +
        " has strides no runtime layout describes. Pass it in the default order or, at four or five dimensions, in channels-last order.");
  }

  // A dimension of zero is what makes a tensor empty, and an empty one needs no
  // memory. Asked this way rather than by multiplying the sizes, because a
  // capsule can claim a shape whose element count wraps a 64 bit product to
  // zero, and the check would then pass the very input it is here to refuse.
  const bool has_elements = std::none_of(
      shape.begin(), shape.end(), [](py::ssize_t size) { return size == 0; });
  if (described.data == nullptr && has_elements) {
    throw std::runtime_error(
        prefix + " has no data, so there is nothing for the method to read.");
  }
  reject_unexpected_layout(
      shape, strides, itemsize, input_index, method_name, expected);
  auto* data = static_cast<uint8_t*>(described.data) + described.byte_offset;
  std::vector<executorch::aten::SizesType> sizes(shape.begin(), shape.end());
  return for_blob(data, std::move(sizes), *scalar_type)
      .dim_order(std::move(*dim_order))
      .dynamism(executorch::aten::TensorShapeDynamism::STATIC)
      .device(device)
      .make_tensor_ptr();
}

#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
// Whether the tensor's bytes run in the order its shape implies, which is what
// a flat read of them assumes.
bool has_default_layout(const executorch::aten::Tensor& tensor) {
  py::ssize_t expected = 1;
  for (ssize_t i = tensor.dim() - 1; i >= 0; --i) {
    if (tensor.size(i) != 1 &&
        static_cast<py::ssize_t>(tensor.strides()[i]) != expected) {
      return false;
    }
    expected *= tensor.size(i);
  }
  return true;
}

/// A format code for raw elements of a given width, for a dtype the buffer
/// protocol cannot name. It keeps the width and the shape, and says nothing
/// about what the bits mean, which is honest: the caller reinterprets them.
const char* unsigned_format_for_width(py::ssize_t itemsize) {
  switch (itemsize) {
    case 2:
      return "H";
    case 4:
      return "I";
    case 8:
      return "Q";
    default:
      return "B";
  }
}

/// The DLPack description of a runtime dtype, which is how bfloat16 and the
/// other dtypes no buffer format code covers are named. One DLPack has no name
/// for is handed over by its width alone.
DLDataType dlpack_dtype(executorch::aten::ScalarType scalar_type) {
  const auto bits =
      static_cast<uint8_t>(executorch::runtime::elementSize(scalar_type) * 8);
  switch (scalar_type) {
    case executorch::aten::ScalarType::Bool:
      return {kDLBool, bits, 1};
    case executorch::aten::ScalarType::Byte:
    case executorch::aten::ScalarType::UInt16:
    case executorch::aten::ScalarType::UInt32:
    case executorch::aten::ScalarType::UInt64:
      return {kDLUInt, bits, 1};
    case executorch::aten::ScalarType::Char:
    case executorch::aten::ScalarType::Short:
    case executorch::aten::ScalarType::Int:
    case executorch::aten::ScalarType::Long:
      return {kDLInt, bits, 1};
    case executorch::aten::ScalarType::Half:
    case executorch::aten::ScalarType::Float:
    case executorch::aten::ScalarType::Double:
      return {kDLFloat, bits, 1};
    case executorch::aten::ScalarType::BFloat16:
      return {kDLBfloat, bits, 1};
    case executorch::aten::ScalarType::ComplexFloat:
    case executorch::aten::ScalarType::ComplexDouble:
      return {kDLComplex, bits, 1};
    default:
      // Not refused: a consumer that does not know this dtype will say so, and
      // one that does can still read it by width.
      return {kDLOpaqueHandle, bits, 1};
  }
}

/// An owned copy of a tensor the runtime produced, readable through CPython's
/// buffer protocol, so `numpy.asarray(output)` needs no copy and no torch. It
/// is a copy rather than a view because output memory is planned at export time
/// and the next execution writes over it.
struct PyTensorBuffer final {
  explicit PyTensorBuffer(const executorch::aten::Tensor& tensor)
      : PyTensorBuffer(tensor, std::vector<uint8_t>(tensor.nbytes())) {
    if (tensor.const_data_ptr() != nullptr) {
      std::memcpy(data_.data(), tensor.const_data_ptr(), data_.size());
    }
  }

  // Takes the bytes the runtime wrote into instead of copying them. Used for an
  // output the method was exported without planned memory for, where the
  // binding owns the buffer the kernels wrote to, so handing it over is free.
  PyTensorBuffer(
      const executorch::aten::Tensor& tensor,
      std::vector<uint8_t>&& data)
      : scalar_type_(tensor.scalar_type()),
        itemsize_(executorch::runtime::elementSize(tensor.scalar_type())),
        shape_(tensor.sizes().begin(), tensor.sizes().end()),
        data_(std::move(data)) {
    // The buffer handed over was sized for the largest shape the method
    // declares, and a dynamically shaped output usually fills less of it, so it
    // is cut down to what this result actually occupies. Otherwise a reader is
    // told there are more elements than were written.
    data_.resize(tensor.nbytes());
    // The strides come from the tensor rather than from its shape, so a
    // channels-last output is presented in the order it is actually stored.
    strides_.reserve(tensor.dim());
    dlpack_strides_.reserve(tensor.dim());
    for (const auto stride : tensor.strides()) {
      strides_.push_back(static_cast<py::ssize_t>(stride) * itemsize_);
      dlpack_strides_.push_back(static_cast<int64_t>(stride));
    }
    dlpack_shape_.assign(tensor.sizes().begin(), tensor.sizes().end());
  }

  void* data() {
    return data_.data();
  }

  py::buffer_info buffer_info() {
    // A dtype with no format code, bfloat16 above all, still has a width and a
    // shape. Presenting an unsigned integer of that width keeps both, so a
    // reader sees the right elements in the right places and only has to
    // reinterpret what each one means. Presenting a flat run of bytes instead
    // would hand back a different shape than the result has, and say nothing
    // about it.
    const auto format = buffer_format(scalar_type_)
                            .value_or(unsigned_format_for_width(itemsize_));
    return py::buffer_info(
        data(),
        itemsize_,
        format,
        static_cast<py::ssize_t>(shape_.size()),
        shape_,
        strides_);
  }

  /// Lends this memory to any library that speaks DLPack, which is how array
  /// libraries hand each other memory. Unlike the buffer protocol it can name
  /// every dtype, including the ones with no format code.
  py::capsule dlpack(py::object self, py::object /*stream*/) {
    auto managed = std::make_unique<DLManagedTensor>();
    managed->dl_tensor.data = data();
    managed->dl_tensor.device = {kDeviceType, kDeviceIndex};
    managed->dl_tensor.ndim = static_cast<int32_t>(shape_.size());
    managed->dl_tensor.dtype = dlpack_dtype(scalar_type_);
    managed->dl_tensor.shape = dlpack_shape_.data();
    managed->dl_tensor.strides = dlpack_strides_.data();
    managed->dl_tensor.byte_offset = 0;
    // The consumer may outlive this object, so it holds a reference for as long
    // as it keeps the capsule, and its deleter is what gives that reference
    // back.
    managed->manager_ctx = self.inc_ref().ptr();
    managed->deleter = [](DLManagedTensor* tensor) {
      py::gil_scoped_acquire acquired;
      py::handle(static_cast<PyObject*>(tensor->manager_ctx)).dec_ref();
      delete tensor;
    };
    // Named "dltensor" so a consumer can claim it by renaming. If nobody does,
    // this destructor runs instead and releases what was reserved above.
    return py::capsule(managed.release(), "dltensor", [](PyObject* capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* tensor = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        if (tensor != nullptr && tensor->deleter != nullptr) {
          tensor->deleter(tensor);
        }
      }
    });
  }

  py::tuple dlpack_device() const {
    return py::make_tuple(kDeviceType, kDeviceIndex);
  }

  py::tuple shape() const {
    py::tuple sizes(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i) {
      sizes[i] = shape_[i];
    }
    return sizes;
  }

  int8_t dtype() const {
    return static_cast<int8_t>(scalar_type_);
  }

  size_t nbytes() const {
    return data_.size();
  }

  std::string repr() const {
    std::string sizes;
    for (size_t i = 0; i < shape_.size(); ++i) {
      sizes += (i == 0 ? "" : ", ") + std::to_string(shape_[i]);
    }
    return "TensorBuffer(shape=(" + sizes +
        "), dtype=" + executorch::runtime::toString(scalar_type_) + ")";
  }

 private:
  /// These bytes are always the host's: output_to_py refuses a result in device
  /// memory before one of these is built.
  static constexpr int32_t kDeviceType = kDLCPU;
  static constexpr int32_t kDeviceIndex = 0;

  executorch::aten::ScalarType scalar_type_;
  py::ssize_t itemsize_;
  std::vector<py::ssize_t> shape_;
  std::vector<py::ssize_t> strides_;
  /// DLPack counts strides in elements and wants its own arrays to point at, so
  /// they are kept beside the ones the buffer protocol uses, which are in
  /// bytes.
  std::vector<int64_t> dlpack_shape_;
  std::vector<int64_t> dlpack_strides_;
  std::vector<uint8_t> data_;
};

// The torch dtypes that match a runtime scalar type, by the name torch itself
// prints. Names rather than objects because this build has no torch headers,
// and one table decides both directions so an input and an output of one dtype
// cannot disagree.
const std::vector<std::pair<const char*, executorch::aten::ScalarType>>&
torch_dtype_names() {
  static const std::vector<std::pair<const char*, executorch::aten::ScalarType>>
      kNames = {
          {"bool", executorch::aten::ScalarType::Bool},
          {"int8", executorch::aten::ScalarType::Char},
          {"uint8", executorch::aten::ScalarType::Byte},
          {"int16", executorch::aten::ScalarType::Short},
          {"uint16", executorch::aten::ScalarType::UInt16},
          {"int32", executorch::aten::ScalarType::Int},
          {"uint32", executorch::aten::ScalarType::UInt32},
          {"int64", executorch::aten::ScalarType::Long},
          {"uint64", executorch::aten::ScalarType::UInt64},
          {"float16", executorch::aten::ScalarType::Half},
          {"float32", executorch::aten::ScalarType::Float},
          {"float64", executorch::aten::ScalarType::Double},
          {"bfloat16", executorch::aten::ScalarType::BFloat16},
          {"complex64", executorch::aten::ScalarType::ComplexFloat},
          {"complex128", executorch::aten::ScalarType::ComplexDouble},
      };
  return kNames;
}

std::optional<executorch::aten::ScalarType> scalar_type_from_torch_dtype(
    const py::handle& dtype) {
  const std::string name = py::str(dtype);
  for (const auto& [torch_name, scalar_type] : torch_dtype_names()) {
    if (name == std::string("torch.") + torch_name) {
      return scalar_type;
    }
  }
  return std::nullopt;
}

std::optional<const char*> torch_dtype_name(
    executorch::aten::ScalarType scalar_type) {
  for (const auto& [torch_name, candidate] : torch_dtype_names()) {
    if (candidate == scalar_type) {
      return torch_name;
    }
  }
  return std::nullopt;
}

// Describes a torch.Tensor's memory for the runtime without linking torch.
// Every property comes from a plain Python call, the same ones a user would
// make, so there is no ATen type and no torch ABI in this build. bfloat16
// arrives here too, which it cannot through the buffer protocol.
TensorPtr tensor_from_torch(
    const py::handle& tensor,
    size_t input_index,
    const char* method_name,
    const Result<TensorInfo>& expected,
    bool copy) {
  const std::string prefix =
      "Input " + std::to_string(input_index) + " for method " + method_name;
  if (py::cast<bool>(tensor.attr("is_neg")()) ||
      py::cast<bool>(tensor.attr("is_conj")())) {
    // The negation and conjugation are not in the memory, they are a flag torch
    // applies when reading it, so the values here are not the values the caller
    // sees.
    throw std::runtime_error(
        prefix +
        " is a negated or conjugated view, whose memory holds different values than the tensor does. Call .resolve_neg() or .resolve_conj() on it first.");
  }
  // Where the memory lives. A host tensor is the common case and answers in one
  // question, so the rest is only asked when the answer is no. A device tensor
  // is passed by its pointer, which is what the runtime wants, so the only
  // thing to refuse is a device the runtime cannot name.
  const bool on_host = py::cast<bool>(tensor.attr("is_cpu"));
  executorch::aten::Device runtime_device(executorch::aten::DeviceType::CPU, 0);
  // A tensor with no elements has no memory for a device to describe, so it is
  // taken as it is rather than refused for a device it never touches.
  if (!on_host && py::cast<int64_t>(tensor.attr("numel")()) != 0) {
    const py::object device = tensor.attr("device");
    if (py::cast<std::string>(device.attr("type")) != "cuda") {
      throw std::runtime_error(
          prefix + " is on device " + std::string(py::str(device)) +
          ", and only CPU and CUDA tensors can be passed to a method.");
    }
    const py::object index = device.attr("index");
    runtime_device = executorch::aten::Device(
        executorch::aten::DeviceType::CUDA,
        index.is_none()
            ? 0
            : static_cast<executorch::runtime::etensor::DeviceIndex>(
                  py::cast<int64_t>(index)));
  }
  const auto scalar_type = scalar_type_from_torch_dtype(tensor.attr("dtype"));
  if (!scalar_type.has_value()) {
    throw std::runtime_error(
        prefix + " has dtype " + std::string(py::str(tensor.attr("dtype"))) +
        ", which no runtime dtype describes.");
  }
  // Named here rather than left to the runtime, which reports only an error
  // code, and the buffer path says the same thing for the same mistake.
  if (expected.ok() && expected->scalar_type() != *scalar_type) {
    throw std::runtime_error(
        prefix + " is " + executorch::runtime::toString(*scalar_type) +
        ", but the method expects " +
        executorch::runtime::toString(expected->scalar_type()) + ".");
  }
  const auto itemsize = py::cast<py::ssize_t>(tensor.attr("element_size")());
  const auto shape = py::cast<std::vector<py::ssize_t>>(tensor.attr("shape"));
  auto strides = py::cast<std::vector<py::ssize_t>>(tensor.attr("stride")());
  reject_unrepresentable_shape(shape, input_index, method_name);
  // Shape and strides are read separately, so nothing guarantees they agree. A
  // subclass that overrides either one can make them disagree, and reading a
  // stride that is not there is a fault rather than a wrong answer.
  if (strides.size() != shape.size()) {
    throw std::runtime_error(
        prefix + " reports " + std::to_string(shape.size()) +
        " dimensions but " + std::to_string(strides.size()) +
        " strides, so its layout cannot be read.");
  }
  // torch counts strides in elements, the runtime layout check in bytes.
  for (auto& stride : strides) {
    stride *= itemsize;
  }
  auto dim_order = dim_order_from_strides(shape, strides, itemsize);
  if (!dim_order.has_value()) {
    throw std::runtime_error(
        prefix +
        " has strides no runtime layout describes. Call .contiguous() on it, or pass it in channels-last order at four or five dimensions.");
  }
  auto* data = reinterpret_cast<uint8_t*>(
      py::cast<uintptr_t>(tensor.attr("data_ptr")()));
  size_t numel = 1;
  for (const auto size : shape) {
    numel *= static_cast<size_t>(size);
  }
  if (data == nullptr && numel > 0) {
    throw std::runtime_error(
        prefix +
        " has no data, so there is nothing for the method to read. A tensor on the meta device has no memory.");
  }
  reject_unexpected_layout(
      shape, strides, itemsize, input_index, method_name, expected);
  std::vector<executorch::aten::SizesType> sizes(shape.begin(), shape.end());
  if (copy && on_host) {
    return make_tensor_ptr(
        std::move(sizes),
        std::vector<uint8_t>(data, data + numel * itemsize),
        std::move(*dim_order),
        {},
        *scalar_type,
        executorch::aten::TensorShapeDynamism::STATIC);
  }
  return for_blob(data, std::move(sizes), *scalar_type)
      .dim_order(std::move(*dim_order))
      .dynamism(executorch::aten::TensorShapeDynamism::STATIC)
      .device(runtime_device)
      .make_tensor_ptr();
}

// An output goes back as a torch.Tensor when the caller is already using torch,
// so existing callers see no change, and as a TensorBuffer otherwise. torch
// builds that tensor itself, from bytes this object owns, so no copy is made
// and this build still links no torch. Torch is never imported here, only used
// when the caller has already imported it: importing it would load libtorch and
// undo the point of this build.
py::object output_to_py(
    const executorch::aten::Tensor& tensor,
    std::vector<uint8_t>* owned,
    bool as_torch) {
  // Asked once here rather than in each arm below, because every one of them
  // reads these bytes, and the host cannot read memory that is not its own.
  if (!tensor.device().is_cpu()) {
    throw std::runtime_error(
        "This result is in device memory, which the host cannot read. Copy it to the host first.");
  }
  // Built only on the paths that hand it back or read from it. Two of the torch
  // paths below build their own storage instead, and building this up front
  // would copy the result's bytes for nobody.
  const auto to_buffer = [&] {
    return owned != nullptr
        ? py::cast(PyTensorBuffer(tensor, std::move(*owned)))
        : py::cast(PyTensorBuffer(tensor));
  };
  if (!as_torch) {
    return to_buffer();
  }
  const auto dtype_name = torch_dtype_name(tensor.scalar_type());
  if (!dtype_name.has_value()) {
    return to_buffer();
  }
  const py::object torch = py::module_::import("sys").attr("modules")["torch"];
  const py::object dtype = torch.attr(*dtype_name);
  py::tuple shape(tensor.dim());
  for (ssize_t i = 0; i < tensor.dim(); ++i) {
    shape[i] = tensor.size(i);
  }
  // An empty output has no bytes to lend, and torch.frombuffer refuses a buffer
  // of length zero, so the shape alone builds it. The device is named because
  // this result is host memory whatever a caller set as torch's default device,
  // and a tensor built on another device cannot be read back.
  if (tensor.nbytes() == 0) {
    return torch.attr("empty")(
        shape, py::arg("dtype") = dtype, py::arg("device") = "cpu");
  }
  // frombuffer holds the object it reads, which is what keeps these bytes
  // alive, but it can only read a buffer whose bytes run in the order its shape
  // implies.
  if (has_default_layout(tensor)) {
    return torch.attr("frombuffer")(to_buffer(), py::arg("dtype") = dtype)
        .attr("view")(shape);
  }
  // A channels-last output does not, so its bytes go into something that owns
  // them and the layout is applied to that.
  py::tuple strides(tensor.dim());
  for (ssize_t i = 0; i < tensor.dim(); ++i) {
    strides[i] = tensor.strides()[i];
  }
  py::bytearray bytes(
      static_cast<const char*>(tensor.const_data_ptr()), tensor.nbytes());
  return torch.attr("frombuffer")(bytes, py::arg("dtype") = dtype)
      .attr("as_strided")(shape, strides);
}
#endif // EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

/// Whether the results of a method go back as torch tensors. The build that
/// links torch has no other kind of result. The build that does not hands back
/// torch tensors only to a caller who already has torch, and each object that
/// hands results back asks this once when it is built rather than on every
/// call: asked per call the answer could change mid-run under a caller who
/// imported nothing themselves, since anything that needs torch, exir among
/// them, pulls it in.
bool returns_torch_tensors() {
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
  return py::cast<bool>(
      py::module_::import("sys").attr("modules").attr("__contains__")("torch"));
#else
  return true;
#endif // EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
}

void write_data_to_file(const std::string& path, void* buf, size_t size) {
  FILE* f = fopen(path.c_str(), "w+");
  if (!f) {
    throw std::runtime_error(
        "Failed to open file " + path + ": " + strerror(errno));
  }
  size_t num_written = fwrite(buf, 1, size, f);
  if (num_written != size) {
    fclose(f);
    throw std::runtime_error("Failed to write etdump to file " + path);
  }
  int err = fclose(f);
  if (err) {
    throw std::runtime_error(
        "Failed to close etdump file " + path + ": " + strerror(err));
  }
}

void setup_output_storage(
    Method& method,
    const std::vector<Span<uint8_t>>& output_storages) {
  if (output_storages.size() != method.outputs_size()) {
    THROW_IF_ERROR(
        Error::InvalidArgument,
        "number of output storages %zu does not match number of outputs %zu",
        output_storages.size(),
        method.outputs_size());
  }
  for (size_t i = 0; i < output_storages.size(); ++i) {
    if (output_storages[i].size() == 0) {
      // Skip empty output storages, this would happen for non-tensor outputs
      // and memory planned outputs.
      continue;
    }
    Error output_status = method.set_output_data_ptr(
        output_storages[i].data(), output_storages[i].size(), i);
    // We already should be skipping non-tensor outputs, and memory planned
    // outputs so any error is real.
    THROW_IF_ERROR(
        output_status,
        "set_output_data_ptr failed for output %zu with error 0x%" PRIx32,
        i,
        static_cast<uint32_t>(output_status));
  }
}

inline std::unique_ptr<DataLoader> loader_from_buffer(
    const void* ptr,
    size_t ptr_len) {
  return std::make_unique<BufferDataLoader>(ptr, ptr_len);
}

inline std::unique_ptr<DataLoader> loader_from_file(const std::string& path) {
  Result<MmapDataLoader> res = MmapDataLoader::from(
      path.c_str(), MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
  THROW_IF_ERROR(
      res.error(),
      "Failed to create MmapDataLoader from file %s, error: 0x:%" PRIx32,
      path.c_str(),
      static_cast<uint32_t>(res.error()));

  return std::make_unique<MmapDataLoader>(std::move(res.get()));
}

inline std::unique_ptr<Module> load_module_from_buffer(
    const void* ptr,
    size_t ptr_len,
    std::optional<const void*> data_map_ptr,
    std::optional<size_t> data_map_len,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    Program::Verification program_verification) {
  EXECUTORCH_SCOPE_PROF("load_module_from_buffer");
  auto loader = loader_from_buffer(ptr, ptr_len);

  if (data_map_ptr.has_value() && data_map_len.has_value()) {
    auto data_map_loader =
        loader_from_buffer(data_map_ptr.value(), data_map_len.value());
    return std::make_unique<Module>(
        std::move(loader),
        nullptr, // memory_allocator
        nullptr, // temp_allocator
        std::move(event_tracer), // event_tracer
        std::move(data_map_loader)); // data_map_loader
  }

  return std::make_unique<Module>(
      std::move(loader),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      nullptr); // data_map_loader
}

inline std::unique_ptr<Module> load_module_from_file(
    const std::string& program_path,
    std::optional<const std::string>& data_map_path,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    Program::Verification program_verification) {
  EXECUTORCH_SCOPE_PROF("load_module_from_file");

  auto program_loader = loader_from_file(program_path);
  if (data_map_path.has_value()) {
    auto data_map_loader = loader_from_file(data_map_path.value());
    return std::make_unique<Module>(
        std::move(program_loader),
        nullptr, // memory_allocator
        nullptr, // temp_allocator
        std::move(event_tracer), // event_tracer
        std::move(data_map_loader)); // data_map_loader
  }
  return std::make_unique<Module>(
      std::move(program_loader),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      nullptr); // data_map_loader
}

inline std::unique_ptr<Module> load_module_from_buffer_with_data_file(
    const void* ptr,
    size_t ptr_len,
    const std::string& data_map_path,
    std::unique_ptr<runtime::EventTracer> event_tracer,
    Program::Verification program_verification) {
  auto program_loader = loader_from_buffer(ptr, ptr_len);
  auto data_loader = loader_from_file(data_map_path);
  return std::make_unique<Module>(
      std::move(program_loader),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      std::move(data_loader));
}

inline std::unique_ptr<Module> load_module_from_data_loader(
    std::shared_ptr<PyDataLoader> loader,
    std::optional<const std::string> data_map_path,
    std::unique_ptr<runtime::EventTracer> event_tracer) {
  EXECUTORCH_SCOPE_PROF("load_module_from_data_loader");

  if (data_map_path.has_value()) {
    auto data_map_loader = loader_from_file(data_map_path.value());
    return std::make_unique<Module>(
        loader->make_delegating_loader(),
        nullptr, // memory_allocator
        nullptr, // temp_allocator
        std::move(event_tracer), // event_tracer
        std::move(data_map_loader)); // data_map_loader
  }
  return std::make_unique<Module>(
      loader->make_delegating_loader(),
      nullptr, // memory_allocator
      nullptr, // temp_allocator
      std::move(event_tracer), // event_tracer
      nullptr); // data_map_loader
}

inline py::list get_outputs_as_py_list(
    const std::vector<EValue>& outputs,
    bool results_as_torch,
    bool clone_outputs = true) {
  const auto outputs_size = outputs.size();
  py::list list(outputs_size);
  for (size_t i = 0; i < outputs_size; ++i) {
    auto& v = outputs[i];
    if (Tag::None == v.tag) {
      list[i] = py::none();
    } else if (Tag::Int == v.tag) {
      list[i] = py::cast(v.toInt());
    } else if (Tag::Double == v.tag) {
      list[i] = py::cast(v.toDouble());
    } else if (Tag::Bool == v.tag) {
      list[i] = py::cast(v.toBool());
    } else if (Tag::String == v.tag) {
      list[i] = py::cast(std::string(v.toString().data()));
    } else if (Tag::Tensor == v.tag) {
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
      // Always owned, whatever clone_outputs asks for: nothing here can hold a
      // borrow of the arena safely.
      list[i] = output_to_py(v.toTensor(), nullptr, results_as_torch);
#elif defined(USE_ATEN_LIB)
      // Clone so the outputs in python do not share a lifetime with the
      // module object
      if (clone_outputs) {
        list[i] = py::cast(v.toTensor().clone());
      } else {
        list[i] = py::cast(v.toTensor());
      }
#else
      if (clone_outputs) {
        list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()).clone());
      } else {
        list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()));
      }
#endif
    } else {
      ET_ASSERT_UNREACHABLE_MSG("Invalid model output type");
    }
  }
  return list;
}

static constexpr size_t kDEFAULT_BUNDLED_INPUT_POOL_SIZE = 16 * 1024U;

struct PyBundledModule : public BundledModule {
  // Asked when this object is built and then kept, so a caller's result type
  // cannot change under them. Two objects built at different moments may
  // differ, which is intended: each answers for the process it was loaded into.
  bool returns_torch_tensors_ = returns_torch_tensors();
  explicit PyBundledModule(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size)
      : BundledModule(buffer.cast<std::string_view>().data()),
        bundled_program_ptr_(buffer),
        program_ptr_(static_cast<const void*>(
            bundled_program_flatbuffer::GetBundledProgram(
                get_bundled_program_ptr())
                ->program()
                ->data())),
        program_len_(bundled_program_flatbuffer::GetBundledProgram(
                         get_bundled_program_ptr())
                         ->program()
                         ->size()) {}

  static std::unique_ptr<PyBundledModule> load_from_buffer(
      const py::bytes& buffer,
      uint32_t bundled_input_pool_size) {
    return std::make_unique<PyBundledModule>(buffer, bundled_input_pool_size);
  }

  const void* get_bundled_program_ptr() {
    return bundled_program_ptr_.cast<std::string_view>().data();
  }

  const void* get_program_ptr() {
    return program_ptr_;
  }

  size_t get_program_len() {
    return program_len_;
  }

  py::list verify_result_with_bundled_expected_output(
      const std::string& method_name,
      size_t testset_idx,
      double rtol = 1e-5,
      double atol = 1e-8) {
    // Execute the method
    auto result = BundledModule::execute(method_name, testset_idx);
    if (!result.ok()) {
      THROW_IF_ERROR(
          result.error(),
          "Method execution failed with status 0x%" PRIx32,
          static_cast<uint32_t>(result.error()));
    }

    // Convert outputs to py::list
    const auto& outputs = result.get();
    py::list py_outputs =
        get_outputs_as_py_list(outputs, returns_torch_tensors_);

    Error status = BundledModule::verify_method_outputs(
        method_name, testset_idx, rtol, atol);
    THROW_IF_ERROR(
        status,
        "Result verification failed with status %" PRIu32,
        static_cast<uint32_t>(status));
    return py_outputs;
  }

 private:
  // Store the bytes object instead of a raw pointer so that this module will
  // keep the bytes alive.
  const py::bytes bundled_program_ptr_;
  const void* program_ptr_;
  size_t program_len_;
};

// Program points to DataLoader so bundle them up into a struct to ensure that
// it stays alive.
struct ProgramState final {
  // The program is read out of these bytes rather than copied from them, when
  // it was loaded from a buffer. Declared before the loader so it is destroyed
  // after.
  py::object bytes_;
  std::unique_ptr<DataLoader> loader_;
  std::unique_ptr<Program> program_;
  // Owned here rather than by PyProgram, beside the loader it reads
  // through, because a Method borrows the map and outlives the call that
  // loaded it. Both stay alive as long as any method does.
  std::unique_ptr<DataLoader> data_map_loader_;
  std::unique_ptr<FlatTensorDataMap> data_map_;

  explicit ProgramState(
      py::object bytes,
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<Program> program,
      std::unique_ptr<DataLoader> data_map_loader = nullptr,
      std::unique_ptr<FlatTensorDataMap> data_map = nullptr)
      : bytes_(std::move(bytes)),
        loader_(std::move(loader)),
        program_(std::move(program)),
        data_map_loader_(std::move(data_map_loader)),
        data_map_(std::move(data_map)) {}
  ProgramState(const ProgramState&) = delete;
  ProgramState& operator=(const ProgramState&) = delete;
  ProgramState(ProgramState&&) = default;
  ProgramState& operator=(ProgramState&&) = default;
};

/// Expose a subset of TensorInfo information to python.
struct PyTensorInfo final {
  explicit PyTensorInfo(
      std::shared_ptr<Module> module,
      torch::executor::TensorInfo info)
      : module_(std::move(module)), state_(nullptr), info_(info) {}

  explicit PyTensorInfo(
      std::shared_ptr<ProgramState> state,
      torch::executor::TensorInfo info)
      : module_(nullptr), state_(std::move(state)), info_(info) {}

  py::tuple sizes() const {
    const auto shape = info_.sizes();
    py::tuple tup(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      tup[i] = py::cast(shape[i]);
    }
    return tup;
  }

  int8_t dtype() const {
    return static_cast<
        std::underlying_type<executorch::aten::ScalarType>::type>(
        info_.scalar_type());
  }

  bool is_memory_planned() const {
    return info_.is_memory_planned();
  }

  size_t nbytes() const {
    return info_.nbytes();
  }

  std::string repr() const {
    std::string size_str = "[";
    for (const auto& d : info_.sizes()) {
      size_str.append(std::to_string(d));
      size_str.append(", ");
    }
    if (size_str.length() >= 2) {
      // Pop the last two characters (command and space) and add close bracket.
      size_str.pop_back();
      size_str.pop_back();
    }
    size_str.append("]");
    return "TensorInfo(sizes=" + size_str + ", dtype=" +
        std::string(executorch::runtime::toString(info_.scalar_type())) +
        ", is_memory_planned=" +
        (info_.is_memory_planned() ? "True" : "False") +
        ", nbytes=" + std::to_string(info_.nbytes()) + ")";
  }

 private:
  // TensorInfo relies on either a module or program to be alive.
  std::shared_ptr<Module> module_;
  std::shared_ptr<ProgramState> state_;
  torch::executor::TensorInfo info_;
};

/// Expose a subset of MethodMeta information to python.
struct PyMethodMeta final {
  explicit PyMethodMeta(
      std::shared_ptr<Module> module,
      torch::executor::MethodMeta meta)
      : module_(std::move(module)), state_(nullptr), meta_(meta) {}

  explicit PyMethodMeta(
      std::shared_ptr<ProgramState> state,
      torch::executor::MethodMeta meta)
      : module_(nullptr), state_(std::move(state)), meta_(meta) {}

  const char* name() const {
    return meta_.name();
  }

  size_t num_inputs() const {
    return meta_.num_inputs();
  }

  std::unique_ptr<PyTensorInfo> input_tensor_meta(size_t index) const {
    const auto result = meta_.input_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get input tensor meta at %zu", index);
    if (module_) {
      return std::make_unique<PyTensorInfo>(module_, result.get());
    } else {
      return std::make_unique<PyTensorInfo>(state_, result.get());
    }
  }

  size_t num_outputs() const {
    return meta_.num_outputs();
  }

  std::unique_ptr<PyTensorInfo> output_tensor_meta(size_t index) const {
    const auto result = meta_.output_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get output tensor meta at %zu", index);
    if (module_) {
      return std::make_unique<PyTensorInfo>(module_, result.get());
    } else {
      return std::make_unique<PyTensorInfo>(state_, result.get());
    }
  }

  size_t num_attributes() const {
    return meta_.num_attributes();
  }

  std::unique_ptr<PyTensorInfo> attribute_tensor_meta(size_t index) const {
    const auto result = meta_.attribute_tensor_meta(index);
    THROW_INDEX_IF_ERROR(
        result.error(), "Cannot get attribute tensor meta at %zu", index);
    if (module_) {
      return std::make_unique<PyTensorInfo>(module_, result.get());
    } else {
      return std::make_unique<PyTensorInfo>(state_, result.get());
    }
  }

  py::str repr() const {
    py::list input_meta_strs;
    for (size_t i = 0; i < meta_.num_inputs(); ++i) {
      input_meta_strs.append(py::str(input_tensor_meta(i)->repr()));
    }
    py::list output_meta_strs;
    for (size_t i = 0; i < meta_.num_outputs(); ++i) {
      auto output_tag_res = meta_.output_tag(i);
      THROW_INDEX_IF_ERROR(
          output_tag_res.error(), "Cannot get Tag for output at %zu", i);
      if (output_tag_res.get() == Tag::Tensor) {
        output_meta_strs.append(py::str(output_tensor_meta(i)->repr()));
      } else {
        output_meta_strs.append(
            py::str(runtime::tag_to_string(output_tag_res.get())));
      }
    }
    // Add quotes to be more similar to Python's repr for strings.
    py::str format =
        "MethodMeta(name='{}', num_inputs={}, input_tensor_meta={}, num_outputs={}, output_tensor_meta={})";
    return format.format(
        std::string(meta_.name()),
        std::to_string(meta_.num_inputs()),
        input_meta_strs,
        std::to_string(meta_.num_outputs()),
        output_meta_strs);
  }

 private:
  // Must keep the either the Module or Program object alive or else the meta
  // object is invalidated.
  std::shared_ptr<Module> module_;
  std::shared_ptr<ProgramState> state_;
  torch::executor::MethodMeta meta_;
};

struct PyModule final {
  // Asked when this object is built and then kept, so a caller's result type
  // cannot change under them. Two objects built at different moments may
  // differ, which is intended: each answers for the process it was loaded into.
  bool returns_torch_tensors_ = returns_torch_tensors();
  explicit PyModule(
      const py::bytes& buffer,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : program_bytes_(buffer),
        data_map_bytes_(py::cast(data_map_buffer)),
        debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_buffer(
            buffer.cast<std::string_view>().data(),
            py::len(buffer),
            data_map_buffer.has_value()
                ? std::optional<const void*>(
                      data_map_buffer.value().cast<std::string_view>().data())
                : std::nullopt,
            data_map_buffer.has_value()
                ? std::optional<size_t>(py::len(data_map_buffer.value()))
                : std::nullopt,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      const void* ptr,
      size_t ptr_len,
      std::optional<const void*> data_map_ptr,
      std::optional<size_t> data_map_ptr_len,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_buffer(
            ptr,
            ptr_len,
            data_map_ptr,
            data_map_ptr_len,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      const void* ptr,
      size_t ptr_len,
      const std::string& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_buffer_with_data_file(
            ptr,
            ptr_len,
            data_path,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      const std::string& program_path,
      std::optional<const std::string>& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_file(
            program_path,
            data_path,
            setup_event_tracer(enable_etdump, debug_buffer_size),
            program_verification)) {}

  explicit PyModule(
      std::shared_ptr<PyDataLoader> loader,
      std::optional<const std::string> data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0)
      : debug_buffer_size_(debug_buffer_size),
        module_(load_module_from_data_loader(
            std::move(loader),
            data_path,
            setup_event_tracer(enable_etdump, debug_buffer_size))) {}

  PyModule(const PyModule&) = delete;
  PyModule& operator=(const PyModule&) = delete;
  PyModule(PyModule&&) = default;
  PyModule& operator=(PyModule&&) = default;

  // Module is only valid as long as the python buffer is alive.
  static std::unique_ptr<PyModule> load_from_buffer(
      const py::bytes& buffer,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency) {
    return std::make_unique<PyModule>(
        buffer,
        data_map_buffer,
        enable_etdump,
        debug_buffer_size,
        program_verification);
  }

  static std::unique_ptr<PyModule> load_from_file(
      const std::string& program_path,
      std::optional<const std::string>& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::InternalConsistency) {
    return std::make_unique<PyModule>(
        program_path,
        data_path,
        enable_etdump,
        debug_buffer_size,
        program_verification);
  }

  // Load with data as a buffer.
  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      std::optional<const py::bytes> data_map_buffer,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    std::optional<const void*> data_map_ptr = std::nullopt;
    std::optional<size_t> data_map_len = std::nullopt;

    if (data_map_buffer.has_value()) {
      data_map_ptr = data_map_buffer.value().cast<std::string_view>().data();
      data_map_len = py::len(data_map_buffer.value());
    }

    return std::make_unique<PyModule>(
        m.get_program_ptr(),
        m.get_program_len(),
        data_map_ptr,
        data_map_len,
        enable_etdump,
        debug_buffer_size,
        Program::Verification::InternalConsistency);
  }

  // Load with data as a file.
  static std::unique_ptr<PyModule> load_from_bundled_program(
      PyBundledModule& m,
      const std::string& data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    return std::make_unique<PyModule>(
        m.get_program_ptr(),
        m.get_program_len(),
        data_path,
        enable_etdump,
        debug_buffer_size,
        Program::Verification::InternalConsistency);
  }

  // Load from an external data loader.
  // This allows external libraries (like PTEZ) to provide custom data loaders.
  static std::unique_ptr<PyModule> load_from_data_loader(
      std::shared_ptr<PyDataLoader> loader,
      std::optional<const std::string> data_path,
      bool enable_etdump,
      size_t debug_buffer_size = 0) {
    return std::make_unique<PyModule>(
        std::move(loader), data_path, enable_etdump, debug_buffer_size);
  }

  py::list run_method(
      const std::string& method_name,
      const py::object& inputs,
      bool clone_outputs = true) {
    const py::sequence input_sequence = as_input_sequence(inputs);
    const auto inputs_size = py::len(input_sequence);
    const auto meta = module_->method_meta(method_name);
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);
    // The views own their memory for the whole call, since the runtime may
    // share a buffer rather than copy it.
    std::vector<py::buffer_info> input_buffers;
    std::vector<DLPackInput> dlpack_inputs;
    std::vector<TensorPtr> buffer_tensors;
    input_buffers.reserve(inputs_size);
    buffer_tensors.reserve(inputs_size);

#if !defined(USE_ATEN_LIB) && \
    !defined(EXECUTORCH_PYBINDINGS_WITHOUT_TORCH) // Portable mode
    // So the ETensors and their metadata stay in scope for
    // Module->run_method.
    std::vector<torch::executor::TensorImpl> input_tensors;
    std::vector<std::vector<torch::executor::Tensor::SizesType>> input_sizes;
    std::vector<std::vector<torch::executor::Tensor::StridesType>>
        input_strides;
    std::vector<std::vector<torch::executor::Tensor::DimOrderType>>
        input_dim_order;
    // We store pointers to these vector elements so important to reserve so
    // that we don't lose those on a vector resize. Don't need to do this for
    // the others since they are vectors of vectors, and we don't store a
    // pointer to the root level vector data.
    input_tensors.reserve(inputs_size);
#endif

    // Convert python objects into EValues.
    for (size_t i = 0; i < inputs_size; ++i) {
      auto python_input = input_sequence[i];
      const auto expected = meta.ok() ? meta->input_tensor_meta(i)
                                      : Result<TensorInfo>(meta.error());
      if (is_torch_tensor(python_input)) {
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
        buffer_tensors.push_back(tensor_from_torch(
            python_input,
            i,
            method_name.c_str(),
            expected,
            input_needs_owned_copy(
                expected,
                /*pins_memory=*/false)));
        cpp_inputs.push_back(EValue(buffer_tensors.back()));
#else
        auto at_tensor = python_input.cast<at::Tensor>();
        reject_lazy_view(at_tensor, i, method_name.c_str());
        reject_unexpected_layout(at_tensor, i, method_name, expected);
#ifdef USE_ATEN_LIB
        EValue evalue(at_tensor);
#else
        // convert at::Tensor to torch::executor::Tensor
        auto type =
            torch_to_executorch_scalar_type(at_tensor.options().dtype());
        size_t dim = at_tensor.dim();
        reject_unrepresentable_shape(at_tensor.sizes(), i, method_name);
        // cant directly alias at::Tensor sizes and strides due to int64 vs
        // int32 typing conflict
        input_sizes.emplace_back(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        input_strides.emplace_back(
            at_tensor.strides().begin(), at_tensor.strides().end());

        // The layout the runtime can describe for this memory, which is the
        // default order or channels-last and nothing else.
        auto dim_order = tensor_dim_order(at_tensor);
        if (!dim_order.has_value()) {
          auto error_msg = "Input " + std::to_string(i) + " for method " +
              method_name +
              " should be contiguous, or channels-last at four or five dimensions.";
          throw std::runtime_error(error_msg);
        }
        input_dim_order.push_back(std::move(*dim_order));
        // The runtime has two device types, so a device outside that pair
        // cannot be represented at all and the tensor would carry a label that
        // does not describe its memory.
        auto mapped_device = torch_to_executorch_device(at_tensor.device());
        // An empty tensor has no buffer for a label to describe, so it is left
        // alone rather than rejected for a device it never touches.
        if (!mapped_device.has_value() && at_tensor.numel() != 0) {
          throw std::runtime_error(
              "Input " + std::to_string(i) + " for method " + method_name +
              " is on device " + at_tensor.device().str() +
              ", and only CPU and CUDA tensors can be passed to a method.");
        }
        const auto device = mapped_device.value_or(
            torch::executor::Device(torch::executor::DeviceType::CPU));
        reject_tensor_without_data(at_tensor, i, method_name.c_str());
        input_tensors.emplace_back(
            type,
            dim,
            input_sizes.back().data(),
            nullptr,
            input_dim_order.back().data(),
            input_strides.back().data(),
            torch::executor::TensorShapeDynamism::STATIC,
            device.type(),
            device.index());

        torch::executor::Tensor temp =
            torch::executor::Tensor(&input_tensors.back());
        alias_etensor_to_attensor(at_tensor, temp);
        EValue evalue(temp);
#endif // USE_ATEN_LIB

        cpp_inputs.push_back(evalue);
#endif // EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
      } else if (py::isinstance<py::float_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<double>(python_input)));
      } else if (py::isinstance<py::buffer>(python_input)) {
        input_buffers.push_back(py::cast<py::buffer>(python_input).request());
        buffer_tensors.push_back(tensor_from_buffer(
            input_buffers.back(),
            i,
            method_name.c_str(),
            expected,
            input_needs_owned_copy(
                expected,
                /*pins_memory=*/!input_buffers.back().readonly)));
        cpp_inputs.push_back(EValue(buffer_tensors.back()));
      } else if (py::hasattr(python_input, "__dlpack__")) {
        // Anything that speaks DLPack, the same as the method entry point
        // takes.
        dlpack_inputs.emplace_back(
            py::reinterpret_borrow<py::object>(python_input));
        buffer_tensors.push_back(
            tensor_from_dlpack(dlpack_inputs.back(), i, method_name, expected));
        cpp_inputs.push_back(EValue(buffer_tensors.back()));
      } else {
        throw std::runtime_error(
            "Unsupported python type " +
            std::string(py::str(python_input.get_type())) +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    // Set up output storage before execution.
    allocate_output_tensors(method_name);
    auto outputs = module_->execute(method_name, cpp_inputs);
    THROW_IF_ERROR(
        outputs.error(),
        "Failed to execute method %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(outputs.error()));

    // Retrieve outputs
    return get_outputs_as_py_list(
        outputs.get(), returns_torch_tensors_, clone_outputs);
  }

  py::list forward(const py::object& inputs, bool clone_outputs = true) {
    return run_method("forward", inputs, clone_outputs);
  }

  bool has_etdump() {
    ETDumpGen* etdump = dynamic_cast<ETDumpGen*>(module_->event_tracer());
    return etdump != nullptr;
  }

  void write_etdump_result_to_file(
      const std::string& path,
      const py::object& debug_buffer_path) {
    if (!has_etdump()) {
      throw std::runtime_error("No etdump found");
    }
    ETDumpGen* etdump = dynamic_cast<ETDumpGen*>(module_->event_tracer());
    etdump_result result = etdump->get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      write_data_to_file(path, result.buf, result.size);
      free(result.buf);
      if (py::isinstance<py::str>(debug_buffer_path)) {
        // Also write out the debug buffer to a separate file if requested.
        std::string debug_buffer_path_str =
            py::cast<std::string>(debug_buffer_path);
        if (debug_buffer_ && debug_buffer_size_ > 0) {
          write_data_to_file(
              debug_buffer_path_str, debug_buffer_.get(), debug_buffer_size_);
        }
      }
    } else {
      ET_LOG(
          Info,
          "No etdump data found, try rebuilding with "
          "the CMake option EXECUTORCH_ENABLE_EVENT_TRACER or with "
          "buck run --config executorch.event_tracer_enabled=true");
    }
  }

  py::list plan_execute(
      const std::string method_name,
      bool clone_outputs = true) {
    auto status = module_->load_method(method_name);

    THROW_IF_ERROR(
        status,
        "executing execution plan for method 'load' failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(status));
    auto output = module_->execute(method_name.c_str());
    THROW_IF_ERROR(
        output.error(),
        "executing execution plan for method 'forward' failed with error: 0x%" PRIx32,
        static_cast<uint32_t>(output.error()));
    return get_outputs_as_py_list(
        output.get(), returns_torch_tensors_, clone_outputs);
  }

  std::unique_ptr<PyMethodMeta> method_meta(const std::string method_name) {
    auto method_data = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_data.error(),
        "failed to retrieve method_meta for method %s, error 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(method_data.error()));
    return std::make_unique<PyMethodMeta>(module_, method_data.get());
  }

  std::vector<std::string> method_names() {
    auto result = module_->method_names();
    THROW_IF_ERROR(
        result.error(),
        "Failed to get method names, error: 0x%" PRIx32,
        static_cast<uint32_t>(result.error()));
    const auto& method_set = result.get();
    return std::vector<std::string>(method_set.begin(), method_set.end());
  }

 private:
  // The program is read straight out of these bytes rather than copied, so they
  // have to outlive the module. Declared before it so they are destroyed after.
  py::object program_bytes_;
  py::object data_map_bytes_;
  // Hold onto the debug_buffer_ for the event_tracer.
  std::unique_ptr<uint8_t[]> debug_buffer_;
  size_t debug_buffer_size_;

  std::shared_ptr<Module> module_;
  // Need to keep-alive output tensors until they can be compared in case of
  // bundled programs.
  std::vector<std::optional<TensorPtr>> output_tensors_;

  // Set debug buffer for potential event tracer.
  std::unique_ptr<torch::executor::ETDumpGen> setup_event_tracer(
      bool enable_etdump,
      size_t debug_buffer_size) {
    std::unique_ptr<torch::executor::ETDumpGen> event_tracer = enable_etdump
        ? std::make_unique<torch::executor::ETDumpGen>()
        : nullptr;
    if (enable_etdump && debug_buffer_size > 0) {
      debug_buffer_ = std::make_unique<uint8_t[]>(debug_buffer_size);
      debug_buffer_size_ = debug_buffer_size;
      event_tracer->set_debug_buffer(
          Span<uint8_t>(debug_buffer_.get(), debug_buffer_size));
      event_tracer->set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kIntermediateOutputs);
    }
    return event_tracer;
  }

  // Allocate output tensors when they are not memory planned.
  void allocate_output_tensors(const std::string& method_name) {
    auto method_meta_result = module_->method_meta(method_name);
    THROW_IF_ERROR(
        method_meta_result.error(),
        "Failed to get method_meta for %s, error: 0x%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(method_meta_result.error()));

    auto method_meta = method_meta_result.get();
    const auto num_outputs = method_meta.num_outputs();

    // Create a buffer for each output tensor. Memory planned outputs and non
    // tensor outputs get an empty buffer in this list which is ignored later.
    output_tensors_.clear();
    output_tensors_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      auto output_type = method_meta.output_tag(i);
      THROW_IF_ERROR(
          output_type.error(), "Failed to get output type for output %zu", i);
      if (output_type.get() != Tag::Tensor) {
        // Skip allocating storage for non-tensor outputs.
        output_tensors_.emplace_back(std::nullopt);
        continue;
      }
      const auto& output_tensor_meta = method_meta.output_tensor_meta(i);
      THROW_IF_ERROR(
          output_tensor_meta.error(),
          "Failed to get output tensor meta for output %zu",
          i);
      if (output_tensor_meta.get().is_memory_planned()) {
        // Skip allocating storage for planned memory outputs.
        output_tensors_.emplace_back(std::nullopt);
        continue;
      }
      TensorPtr tensor_ptr = make_tensor_ptr(
          std::vector<executorch::aten::SizesType>(
              output_tensor_meta->sizes().begin(),
              output_tensor_meta->sizes().end()),
          std::vector<uint8_t>(output_tensor_meta->nbytes()),
          output_tensor_meta->scalar_type());
      output_tensors_.emplace_back(std::move(tensor_ptr));
    }

    for (size_t i = 0; i < output_tensors_.size(); ++i) {
      if (output_tensors_[i].has_value()) {
        // Set output tensors on module.
        auto status = module_->set_output(method_name, output_tensors_[i], i);
        THROW_IF_ERROR(
            status,
            "Failed to set output for method %s, error: 0x%" PRIx32,
            method_name.c_str(),
            static_cast<uint32_t>(status));
      }
    }
  }
};

inline std::shared_ptr<ProgramState> load_program(
    std::unique_ptr<DataLoader> loader,
    Program::Verification program_verification,
    std::optional<const std::string> data_path = std::nullopt,
    py::object bytes = py::none()) {
  Result<Program> res = Program::load(loader.get(), program_verification);
  THROW_IF_ERROR(
      res.error(),
      "Failed to load program, error: 0x:%" PRIx32,
      static_cast<uint32_t>(res.error()));
  // A program whose weights live outside the pte names them through a data
  // map. The CUDA delegate emits exactly that: a pte holding the compiled
  // kernels and a separate file holding the weights, so loading the pte alone
  // produces a program that fails when a method runs rather than when it
  // loads.
  std::unique_ptr<DataLoader> data_map_loader;
  std::unique_ptr<FlatTensorDataMap> data_map;
  if (data_path.has_value()) {
    data_map_loader = loader_from_file(data_path.value());
    Result<FlatTensorDataMap> map_res =
        FlatTensorDataMap::load(data_map_loader.get());
    THROW_IF_ERROR(
        map_res.error(),
        "Failed to load data map from %s, error: 0x:%" PRIx32,
        data_path.value().c_str(),
        static_cast<uint32_t>(map_res.error()));
    data_map = std::make_unique<FlatTensorDataMap>(std::move(map_res.get()));
  }
  return std::make_shared<ProgramState>(
      std::move(bytes),
      std::move(loader),
      std::make_unique<Program>(std::move(res.get())),
      std::move(data_map_loader),
      std::move(data_map));
}

/// A wrapper/util class for executorch memory allocations/manager.
class ProgramMemory {
 public:
  /// `devices` is empty when every buffer is on the host, which keeps
  /// `MemoryManager::has_device_memory()` false for CPU-only programs.
  /// Otherwise it holds one entry per buffer, indexed like `sizes`.
  ///
  /// Members initialize in declaration order and each one reads the members
  /// declared before it, so that order is load-bearing. Device buffers come
  /// first so that a device that is missing or out of memory throws before the
  /// host arenas are allocated and zero-filled, rather than after.
  ProgramMemory(
      std::vector<int64_t>&& sizes,
      std::vector<runtime::etensor::Device>&& devices)
      : runtime_allocator_(),
        planned_sizes_(std::move(sizes)),
        planned_devices_(std::move(devices)),
        device_buffers_(allocate_device_buffers()),
        non_const_buffers_(allocate_host_buffers()),
        non_const_spans_(create_non_const_spans()),
        non_const_allocator_(create_non_const_allocator()),
        mem_manager_(
            &const_allocator_,
            &non_const_allocator_,
            &runtime_allocator_,
            &temp_allocator_) {}

  explicit ProgramMemory(std::vector<int64_t>&& sizes)
      : ProgramMemory(std::move(sizes), {}) {}

  /// Returns a pointer to the internal memory manager, the Memory instance
  /// must outlive this pointer.
  MemoryManager* mem_manager() {
    return &mem_manager_;
  }

  ProgramMemory(const ProgramMemory&) = delete;
  ProgramMemory& operator=(const ProgramMemory&) = delete;

 private:
  MemoryAllocator const_allocator_{MemoryAllocator(0, nullptr)};

  MallocMemoryAllocator runtime_allocator_;

  MallocMemoryAllocator temp_allocator_{};

  std::vector<int64_t> planned_sizes_;

  std::vector<runtime::etensor::Device> planned_devices_;

  // Backs device-tagged buffers; the entry is empty for a CPU-tagged buffer.
  // Parallel to non_const_buffers_ so both index by planned buffer id. Empty
  // for an all-host program.
  std::vector<DeviceMemoryBuffer> device_buffers_;

  // Backs CPU-tagged buffers; the entry is empty for a device-tagged buffer.
  std::vector<std::vector<uint8_t>> non_const_buffers_;

  std::vector<Span<uint8_t>> non_const_spans_;

  HierarchicalAllocator non_const_allocator_;

  MemoryManager mem_manager_;

  bool is_device_buffer(size_t index) const {
    return index < planned_devices_.size() && !planned_devices_[index].is_cpu();
  }

  std::vector<std::vector<uint8_t>> allocate_host_buffers() {
    std::vector<std::vector<uint8_t>> result;
    result.reserve(planned_sizes_.size());
    for (size_t i = 0; i < planned_sizes_.size(); ++i) {
      if (is_device_buffer(i)) {
        result.emplace_back();
      } else {
        result.emplace_back(planned_sizes_[i]);
      }
    }
    return result;
  }

  std::vector<DeviceMemoryBuffer> allocate_device_buffers() {
    std::vector<DeviceMemoryBuffer> result;
    if (planned_devices_.empty()) {
      return result;
    }
    // Both vectors are filled in lockstep today, so this only fires if a
    // future caller breaks that. HierarchicalAllocator aborts on a mismatch,
    // so check here instead, where a Python caller can catch it.
    THROW_IF_ERROR(
        planned_devices_.size() == planned_sizes_.size()
            ? Error::Ok
            : Error::InvalidArgument,
        "Have %zu planned buffer sizes but %zu device tags",
        planned_sizes_.size(),
        planned_devices_.size());
    result.reserve(planned_sizes_.size());
    for (size_t i = 0; i < planned_sizes_.size(); ++i) {
      if (!is_device_buffer(i)) {
        result.emplace_back();
        continue;
      }
      auto buffer = DeviceMemoryBuffer::create(
          planned_sizes_[i],
          planned_devices_[i].type(),
          planned_devices_[i].index());
      THROW_IF_ERROR(
          buffer.error(),
          "Failed to allocate %" PRId64 " bytes for buffer %zu on device %d:%d",
          planned_sizes_[i],
          i,
          static_cast<int>(planned_devices_[i].type()),
          static_cast<int>(planned_devices_[i].index()));
      result.emplace_back(std::move(buffer.get()));
    }
    return result;
  }

  std::vector<Span<uint8_t>> create_non_const_spans() {
    std::vector<Span<uint8_t>> result;
    result.reserve(planned_sizes_.size());
    for (size_t i = 0; i < planned_sizes_.size(); ++i) {
      if (is_device_buffer(i)) {
        result.push_back(device_buffers_[i].as_span());
      } else {
        result.push_back(
            {non_const_buffers_[i].data(), non_const_buffers_[i].size()});
      }
    }
    return result;
  }

  HierarchicalAllocator create_non_const_allocator() {
    Span<Span<uint8_t>> buffers(
        non_const_spans_.data(), non_const_spans_.size());
    return planned_devices_.empty()
        ? HierarchicalAllocator(buffers)
        : HierarchicalAllocator(
              buffers, {planned_devices_.data(), planned_devices_.size()});
  }
};

/// True if any of the method's memory-planned buffers must live off the host.
bool has_device_buffers(const MethodMeta& method_meta) {
  for (size_t i = 0; i < method_meta.num_memory_planned_buffers(); ++i) {
    auto device = method_meta.memory_planned_buffer_device(i);
    THROW_IF_ERROR(
        device.error(), "Failed to get device of planned buffer %zu", i);
    if (!device.get().is_cpu()) {
      return true;
    }
  }
  return false;
}

/// Arenas sized and placed for a single method, used when that method's
/// buffers cannot come from the program-wide host arenas. Returns nullptr when
/// every buffer is on the host, so one pass over the metadata answers both
/// whether the method needs its own arenas and how big they are.
std::shared_ptr<ProgramMemory> make_method_memory(
    const MethodMeta& method_meta) {
  const size_t num_buffers = method_meta.num_memory_planned_buffers();
  std::vector<int64_t> sizes;
  std::vector<runtime::etensor::Device> devices;
  sizes.reserve(num_buffers);
  devices.reserve(num_buffers);
  bool needs_device_memory = false;
  for (size_t i = 0; i < num_buffers; ++i) {
    auto size = method_meta.memory_planned_buffer_size(i);
    THROW_IF_ERROR(size.error(), "Failed to get size of planned buffer %zu", i);
    auto device = method_meta.memory_planned_buffer_device(i);
    THROW_IF_ERROR(
        device.error(), "Failed to get device of planned buffer %zu", i);
    needs_device_memory |= !device.get().is_cpu();
    sizes.push_back(size.get());
    devices.push_back(device.get());
  }
  if (!needs_device_memory) {
    return nullptr;
  }
  return std::make_shared<ProgramMemory>(std::move(sizes), std::move(devices));
}

struct PyMethod final {
  // Asked when this object is built and then kept, so a caller's result type
  // cannot change under them. Two objects built at different moments may
  // differ, which is intended: each answers for the process it was loaded into.
  bool returns_torch_tensors_ = returns_torch_tensors();
  explicit PyMethod(
      std::shared_ptr<ProgramMemory> memory,
      std::shared_ptr<ProgramState> state,
      std::unique_ptr<Method> method)
      : memory_(std::move(memory)),
        state_(std::move(state)),
        method_(std::move(method)) {}

  void set_inputs(const py::object& inputs) {
    const InUse guard(*this);
    const py::sequence input_sequence = as_input_sequence(inputs);
    const auto inputs_size = py::len(input_sequence);
    std::vector<EValue> cpp_inputs;
    cpp_inputs.reserve(inputs_size);
    // The inputs a previous call was refused for are still here, so the runtime
    // was never left pointing at freed memory. Nothing can read them, because
    // execute() refuses while the input set is incomplete, so they go now
    // rather than piling up over a retry loop.
    if (inputs_incomplete_) {
      input_buffers_.clear();
      input_buffer_tensors_.clear();
      input_objects_.clear();
      borrowed_inputs_.clear();
      dlpack_inputs_.clear();
    }
    // Collected here and kept on the object only once the runtime has accepted
    // them, because a refused call must leave the previous inputs, and the
    // references that keep their memory alive, exactly as they were. A method
    // exported without planned input memory is still pointing at them.
    std::vector<py::buffer_info> buffers;
    std::vector<TensorPtr> buffer_tensors;
    std::vector<py::object> objects;
    std::vector<BorrowedInput> borrowed;
    std::vector<DLPackInput> dlpack_inputs;
    buffers.reserve(inputs_size);
    buffer_tensors.reserve(inputs_size);
    objects.reserve(inputs_size);

    // Convert python objects into EValues.
    for (size_t i = 0; i < inputs_size; ++i) {
      auto python_input = input_sequence[i];
      if (is_torch_tensor(python_input)) {
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
        const auto method_meta = method_->method_meta();
        const auto expected = method_meta.input_tensor_meta(i);
        objects.push_back(py::reinterpret_borrow<py::object>(python_input));
        buffer_tensors.push_back(tensor_from_torch(
            python_input,
            i,
            method_meta.name(),
            expected,
            input_needs_owned_copy(expected, /*pins_memory=*/false)));
        cpp_inputs.push_back(EValue(buffer_tensors.back()));
#else
        auto at_tensor = python_input.cast<at::Tensor>();
        reject_lazy_view(at_tensor, i, method_->method_meta().name());
        reject_unexpected_layout(
            at_tensor,
            i,
            method_->method_meta().name(),
            method_->method_meta().input_tensor_meta(i));

#ifdef USE_ATEN_LIB
        // The same rule the portable arm below applies, for the same reason:
        // the runtime keeps this pointer until execute(), and a tensor object
        // can be resized or have its storage freed while the object lives, so
        // an input the runtime does not copy for itself gets a copy the
        // bindings own.
        if (at_tensor.is_cpu() &&
            input_needs_owned_copy(
                method_->method_meta().input_tensor_meta(i),
                /*pins_memory=*/false)) {
          at_tensor = at_tensor.clone();
        }
        // Held with the buffer inputs, not in a local, because the runtime may
        // keep this pointer until execute() and a local dies at the end of this
        // call.
        objects.push_back(py::reinterpret_borrow<py::object>(python_input));
        buffer_tensors.push_back(
            std::make_shared<executorch::aten::Tensor>(at_tensor));
        EValue evalue(buffer_tensors.back());
#else
        // convert at::Tensor to torch::executor::Tensor
        auto type =
            torch_to_executorch_scalar_type(at_tensor.options().dtype());
        reject_unrepresentable_shape(
            at_tensor.sizes(), i, method_->method_meta().name());
        // cant directly alias at::Tensor sizes and strides due to int64 vs
        // int32 typing conflict
        std::vector<int> sizes(
            at_tensor.sizes().begin(), at_tensor.sizes().end());
        std::vector<int> strides(
            at_tensor.strides().begin(), at_tensor.strides().end());

        // The layout the runtime can describe for this memory, which is the
        // default order or channels-last and nothing else.
        auto dim_order = tensor_dim_order(at_tensor);
        if (!dim_order.has_value()) {
          auto error_msg = "Input " + std::to_string(i) + " for method " +
              method_->method_meta().name() +
              " should be contiguous, or channels-last at four or five dimensions.";
          throw std::runtime_error(error_msg);
        }
        // Record where the buffer actually lives. The conversion copied every
        // other property and left the device at CPU, so an accelerator buffer
        // was described as host memory.
        auto mapped_device = torch_to_executorch_device(at_tensor.device());
        // An empty tensor has no buffer for a label to describe, so it is left
        // alone rather than rejected for a device it never touches.
        if (!mapped_device.has_value() && at_tensor.numel() != 0) {
          throw std::runtime_error(
              "Input " + std::to_string(i) + " for method " +
              method_->method_meta().name() + " is on device " +
              at_tensor.device().str() +
              ", and only CPU and CUDA tensors can be passed to a method.");
        }
        const auto device =
            mapped_device.value_or(aten::Device(aten::DeviceType::CPU));
        reject_tensor_without_data(at_tensor, i, method_->method_meta().name());
        TensorPtr tensor = for_blob(
                               mutable_tensor_data_ptr_no_cow(at_tensor),
                               std::move(sizes),
                               type)
                               .strides(std::move(strides))
                               .dim_order(std::move(*dim_order))
                               .dynamism(aten::TensorShapeDynamism::STATIC)
                               .device(device)
                               .make_tensor_ptr();
        if (at_tensor.is_cpu() &&
            input_needs_owned_copy(
                method_->method_meta().input_tensor_meta(i),
                /*pins_memory=*/false)) {
          // The runtime keeps this pointer, and a tensor cannot promise its
          // memory stays where it is, so it gets a copy the bindings own.
          const auto* bytes =
              static_cast<const uint8_t*>(tensor->const_data_ptr());
          tensor = make_tensor_ptr(
              std::vector<executorch::aten::SizesType>(
                  tensor->sizes().begin(), tensor->sizes().end()),
              std::vector<uint8_t>(bytes, bytes + tensor->nbytes()),
              std::vector<executorch::aten::DimOrderType>(
                  tensor->dim_order().begin(), tensor->dim_order().end()),
              {},
              tensor->scalar_type(),
              aten::TensorShapeDynamism::STATIC);
        }
        // Held with the buffer inputs, not in a local, because the runtime may
        // keep this pointer until execute() and a local dies at the end of this
        // call.
        objects.push_back(py::reinterpret_borrow<py::object>(python_input));
        buffer_tensors.push_back(tensor);
        EValue evalue(buffer_tensors.back());
#endif // USE_ATEN_LIB

        cpp_inputs.push_back(evalue);
#endif // EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
      } else if (py::isinstance<py::none>(python_input)) {
        cpp_inputs.push_back(EValue());
      } else if (py::isinstance<py::bool_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<bool>(python_input)));
      } else if (py::isinstance<py::int_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<int64_t>(python_input)));
      } else if (py::isinstance<py::float_>(python_input)) {
        cpp_inputs.push_back(EValue(py::cast<double>(python_input)));
      } else if (py::isinstance<py::buffer>(python_input)) {
        const auto method_meta = method_->method_meta();
        const auto expected = method_meta.input_tensor_meta(i);
        buffers.push_back(py::cast<py::buffer>(python_input).request());
        const bool copied = input_needs_owned_copy(
            expected, /*pins_memory=*/!buffers.back().readonly);
        buffer_tensors.push_back(tensor_from_buffer(
            buffers.back(), i, method_meta.name(), expected, copied));
        if (!copied) {
          // No copy was made here, so remember where the memory was and check
          // it before running.
          borrowed.push_back(
              {py::reinterpret_borrow<py::object>(python_input),
               buffers.back().ptr,
               static_cast<size_t>(
                   buffers.back().size * buffers.back().itemsize)});
        }
        cpp_inputs.push_back(EValue(buffer_tensors.back()));
      } else if (py::hasattr(python_input, "__dlpack__")) {
        // Anything that speaks DLPack, which is how array libraries hand each
        // other memory, including memory the host cannot read.
        const auto method_meta = method_->method_meta();
        dlpack_inputs.emplace_back(
            py::reinterpret_borrow<py::object>(python_input));
        buffer_tensors.push_back(tensor_from_dlpack(
            dlpack_inputs.back(),
            i,
            method_meta.name(),
            method_meta.input_tensor_meta(i)));
        cpp_inputs.push_back(EValue(buffer_tensors.back()));
      } else {
        throw std::runtime_error(
            "Unsupported python type " +
            std::string(py::str(python_input.get_type())) +
            ". Ensure that inputs are passed as a flat list of tensors.");
      }
    }

    executorch::aten::ArrayRef<EValue> input_evalue_list(
        cpp_inputs.data(), cpp_inputs.size());

    Error set_inputs_status = method_->set_inputs(input_evalue_list);
    if (set_inputs_status != Error::Ok) {
      // The runtime installs inputs one at a time, so a refusal partway through
      // leaves it holding some of these pointers and some of the previous ones.
      // Both sets are kept until a call succeeds and replaces them outright.
      std::move(
          buffers.begin(), buffers.end(), std::back_inserter(input_buffers_));
      std::move(
          buffer_tensors.begin(),
          buffer_tensors.end(),
          std::back_inserter(input_buffer_tensors_));
      std::move(
          objects.begin(), objects.end(), std::back_inserter(input_objects_));
      std::move(
          borrowed.begin(),
          borrowed.end(),
          std::back_inserter(borrowed_inputs_));
      // A capsule's deleter tells its producer the memory can go, so running it
      // here would free what the runtime was just handed.
      std::move(
          dlpack_inputs.begin(),
          dlpack_inputs.end(),
          std::back_inserter(dlpack_inputs_));
      // The runtime may have installed some of them before it refused, and does
      // not say how far it got, so the set is taken as a mix of two calls and
      // must not be executed.
      inputs_incomplete_ = true;
      THROW_IF_ERROR(
          set_inputs_status,
          "method->set_inputs() for method '%s' failed with error 0x%" PRIx32,
          method_->method_meta().name(),
          static_cast<uint32_t>(set_inputs_status));
    }
    inputs_incomplete_ = false;

    input_buffers_ = std::move(buffers);
    input_buffer_tensors_ = std::move(buffer_tensors);
    input_objects_ = std::move(objects);
    borrowed_inputs_ = std::move(borrowed);
    dlpack_inputs_ = std::move(dlpack_inputs);
  }

  void execute() {
    const InUse guard(*this);
    if (inputs_incomplete_) {
      throw std::runtime_error(
          "The last set_inputs was refused partway through, so this method holds some inputs from that call and some from the one before. Set all of them again before executing.");
    }
    // An execution that does not finish leaves the outputs holding neither this
    // call's results nor, once the storages below are replaced, the last one's.
    executed_ = false;
    reject_moved_inputs();
    const auto num_outputs = method_->outputs_size();
    output_objects_.clear();
    allocate_output_storages();
    std::vector<Span<uint8_t>> output_storage_spans(num_outputs);
    for (int i = 0; i < output_storages_.size(); ++i) {
      output_storage_spans[i] =
          Span<uint8_t>(output_storages_[i].data(), output_storages_[i].size());
    }
#ifdef USE_ATEN_LIB
    // [TLS handling] This is to workaround an assertion failure
    // (https://fburl.com/code/302jyn8d) running `gelu` in ATen mode in fbcode
    // (such as bento). The problem is ExecuTorch ATen mode doesn't have
    // Thread Local State, but `torch-cpp` is assuming tls init is done. There
    // are two more checks: MKLDNN disabled and C10_MOBILE, if any of them is
    // true we won't be hitting this assertion error. However in `torch-cpp`
    // lib both checks are false. Production impact: this should not make any
    // impact in production environment, given that in xplat we are depending
    // on a library that enables C10_MOBILE (`torch_mobile_core`).
    c10::impl::ExcludeDispatchKeyGuard no_autograd(
        c10::autograd_dispatch_keyset);
#endif
    setup_output_storage(*method_, output_storage_spans);
    Error execute_status = method_->execute();
    THROW_IF_ERROR(
        execute_status,
        "method->execute() failed with error 0x%" PRIx32,
        static_cast<uint32_t>(execute_status));
    executed_ = true;
  }

  py::list get_outputs(bool clone_outputs = true) {
    const InUse guard(*this);
    if (inputs_incomplete_) {
      throw std::runtime_error(
          "This method's inputs were left incomplete by a refused call, so it has no outputs to hand back. Set all of them again and execute before reading the outputs.");
    }
    if (!executed_) {
      // An output the method was exported without planned memory for has no
      // memory at all until an execution gives it some, and one in the arena
      // holds whatever was there before, so there is nothing to hand back.
      throw std::runtime_error(
          "This method has not completed an execution, so it has no outputs to hand back. Execute it first.");
    }
    std::vector<EValue> result(method_->outputs_size());

    Error get_outputs_status =
        method_->get_outputs(result.data(), method_->outputs_size());
    THROW_IF_ERROR(
        get_outputs_status,
        "method->get_outputs() for method '%s' failed with error 0x%" PRIx32,
        method_->method_meta().name(),
        static_cast<uint32_t>(get_outputs_status));

    // Retrieve outputs
    return outputs_to_py_list(result, clone_outputs);
  }

  py::list call(const py::object& inputs, bool clone_outputs = true) {
    const InUse guard(*this);
    set_inputs(inputs);
    execute();
    return get_outputs(clone_outputs);
  }

  py::object get_attribute(const std::string& name) {
    Result<executorch::aten::Tensor> attr = method_->get_attribute(name);
    THROW_IF_ERROR(
        attr.error(),
        "Failed to get attribute '%s' for method '%s', error: 0x:%" PRIx32,
        name.c_str(),
        method_->method_meta().name(),
        static_cast<uint32_t>(attr.error()));
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
    return output_to_py(attr.get(), nullptr, returns_torch_tensors_);
#elif defined(USE_ATEN_LIB)
    return py::cast(attr.get());
#else
    return py::cast(alias_attensor_to_etensor(attr.get()));
#endif
  }

  PyMethodMeta method_meta() {
    return PyMethodMeta(state_, method_->method_meta());
  }

 private:
  // Method keeps a reference to the memory manager, so we need to keep this
  // alive
  std::shared_ptr<ProgramMemory> memory_;
  // Method keeps a reference to the program, so we also need to keep this alive
  std::shared_ptr<ProgramState> state_;
  std::unique_ptr<Method> method_;
  // Need to keep-alive output storages until they can be compared in case of
  // bundled programs.
  std::vector<std::vector<uint8_t>> output_storages_;
  /// For each input the runtime was given a pointer to rather than a copy of,
  /// the object it came from and where its memory was at the time. The buffer
  /// protocol says an exporter must keep that memory in place while a view is
  /// held, and Python enforces it for well behaved exporters, but numpy offers
  /// an explicit way out of the promise, so the claim is verified rather than
  /// trusted.
  struct BorrowedInput {
    py::object source;
    const void* data;
    size_t nbytes;
  };
  std::vector<BorrowedInput> borrowed_inputs_;
  /// Capsules claimed from DLPack producers. Held for as long as the runtime
  /// may read them, and given back when they are replaced.
  std::vector<DLPackInput> dlpack_inputs_;
  /// Buffer inputs may be shared rather than copied, so their views and the
  /// tensors that own any copied bytes are held from set_inputs() until the
  /// next one replaces them.
  std::vector<py::buffer_info> input_buffers_;
  std::vector<TensorPtr> input_buffer_tensors_;
  std::vector<py::object> input_objects_;
  // A method holds the inputs it was given and the outputs it handed back, so
  // two callers cannot be inside one at the same time. Converting an output
  // calls into torch, which can let another thread run, so this is refused
  // rather than interleaved.
  std::thread::id user_;
  size_t depth_ = 0;
  bool inputs_incomplete_ = false;
  // Whether an execution has run to completion since the last one started. The
  // outputs are only readable then: before it, they are memory nothing has
  // written, and after a refused one, memory the refusal may have released.
  bool executed_ = false;

  class InUse final {
   public:
    explicit InUse(PyMethod& method) : method_(method) {
      const auto self = std::this_thread::get_id();
      if (method_.depth_ > 0 && method_.user_ != self) {
        throw std::runtime_error(
            "This method is already running on another thread. A method keeps the inputs and outputs of one call, so it cannot be shared. Load a method per thread.");
      }
      method_.user_ = self;
      ++method_.depth_;
    }
    ~InUse() {
      --method_.depth_;
    }

   private:
    PyMethod& method_;
  };
  // What this execution already handed back. An unplanned output is given away
  // rather than copied, and the method still points at those bytes, so a second
  // look has to return the same object instead of reading the memory again.
  std::vector<py::object> output_objects_;

  void allocate_output_storages() {
    const auto num_outputs = method_->outputs_size();
    // Create a buffer for each output tensor. Memory planned outputs and non
    // tensor outputs get an empty buffer in this list which is ignored later.
    output_storages_.resize(num_outputs);
    auto meta = method_->method_meta();
    for (size_t i = 0; i < num_outputs; ++i) {
      auto output_type = meta.output_tag(i);
      THROW_IF_ERROR(
          output_type.error(), "Failed to get output type for output %zu", i);
      if (output_type.get() != Tag::Tensor) {
        // Skip allocating storage for non-tensor outputs.
        continue;
      }
      const auto& output_tensor_meta =
          method_->method_meta().output_tensor_meta(i);
      THROW_IF_ERROR(
          output_tensor_meta.error(),
          "Failed to get output tensor meta for output %zu",
          i);
      if (output_tensor_meta.get().is_memory_planned()) {
        // Skip allocating storage for planned memory outputs.
        continue;
      }
      const auto is_input = meta.output_is_input(i);
      THROW_IF_ERROR(is_input.error(), "Failed to inspect output %zu", i);
      if (is_input.get()) {
        // This output is one of the inputs, so the two share their memory.
        // Giving it a buffer would move the input into that buffer as well, and
        // the method would read whatever the buffer held rather than the input.
        continue;
      }
      // Allocate storage for the output tensor. A storage that was handed to
      // python is left behind empty, so a new one is allocated here rather than
      // written over, and the caller keeps what it was given.
      const size_t output_size = output_tensor_meta.get().nbytes();
      if (output_storages_[i].size() != output_size) {
        output_storages_[i] = std::vector<uint8_t>(output_size);
      }
    }
  }

  /// Whether this output was written into a buffer this object owns, rather
  /// than into the arena. Such a buffer is replaced on the next call.
  bool owns_output_storage(size_t index, const executorch::aten::Tensor& tensor)
      const {
    return index < output_storages_.size() &&
        !output_storages_[index].empty() &&
        output_storages_[index].data() == tensor.const_data_ptr();
  }

  /// Refuses to run when memory the runtime was given has moved since it was
  /// given. Reading it would be reading freed memory, and the answer would look
  /// like numbers rather than a failure.
  void reject_moved_inputs() {
    for (size_t i = 0; i < borrowed_inputs_.size(); ++i) {
      const auto& borrowed = borrowed_inputs_[i];
      const auto current = py::cast<py::buffer>(borrowed.source).request();
      if (current.ptr != borrowed.data ||
          static_cast<size_t>(current.size * current.itemsize) !=
              borrowed.nbytes) {
        borrowed_inputs_.clear();
        input_buffers_.clear();
        input_buffer_tensors_.clear();
        input_objects_.clear();
        inputs_incomplete_ = true;
        throw std::runtime_error(
            "The memory behind one of the inputs moved after it was set, so this method is pointing at memory that is no longer there. Do not resize or reallocate something you have passed in. Set the inputs again before executing.");
      }
    }
  }

  py::list outputs_to_py_list(
      const std::vector<EValue>& outputs,
      bool clone_outputs = true) {
    const auto outputs_size = outputs.size();
    py::list list(outputs_size);
    for (size_t i = 0; i < outputs_size; ++i) {
      auto& v = outputs[i];
      if (Tag::None == v.tag) {
        list[i] = py::none();
      } else if (Tag::Int == v.tag) {
        list[i] = py::cast(v.toInt());
      } else if (Tag::Double == v.tag) {
        list[i] = py::cast(v.toDouble());
      } else if (Tag::Bool == v.tag) {
        list[i] = py::cast(v.toBool());
      } else if (Tag::String == v.tag) {
        list[i] = py::cast(std::string(v.toString().data()));
      } else if (Tag::Tensor == v.tag) {
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
        // An output the method was exported without planned memory for was
        // written straight into a buffer this object owns, so that buffer is
        // handed over as it is. A planned output lives in the arena, which the
        // next execution writes over, so it is copied.
        output_objects_.resize(outputs_size);
        if (output_objects_[i]) {
          list[i] = output_objects_[i];
          continue;
        }
        const bool owns_it = owns_output_storage(i, v.toTensor());
        output_objects_[i] = output_to_py(
            v.toTensor(),
            owns_it ? &output_storages_[i] : nullptr,
            returns_torch_tensors_);
        list[i] = output_objects_[i];
#elif defined(USE_ATEN_LIB)
        // An output written into a buffer this object owns cannot be handed out
        // by reference, because that buffer is replaced on the next call and a
        // reference to it would be reading memory that has been freed. A
        // planned output lives in the arena, which lasts as long as the method,
        // so asking for no copy there still means no copy.
        if (clone_outputs || owns_output_storage(i, v.toTensor())) {
          list[i] = py::cast(v.toTensor().clone());
        } else {
          list[i] = py::cast(v.toTensor());
        }
#else
        if (clone_outputs || owns_output_storage(i, v.toTensor())) {
          list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()).clone());
        } else {
          list[i] = py::cast(alias_attensor_to_etensor(v.toTensor()));
        }
#endif
      } else {
        ET_ASSERT_UNREACHABLE_MSG("Invalid model output type");
      }
    }
    return list;
  }
};

struct PyProgram final {
  // Asked when this object is built and then kept, so a caller's result type
  // cannot change under them. Two objects built at different moments may
  // differ, which is intended: each answers for the process it was loaded into.
  bool returns_torch_tensors_ = returns_torch_tensors();
  explicit PyProgram(
      std::unique_ptr<DataLoader> loader,
      std::unique_ptr<ETDumpGen> tracer = nullptr,
      size_t debug_buffer_size = 0,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt,
      // The bytes a program loaded from a buffer is read out of, if any. They
      // are not copied, so they have to outlive the program.
      py::object owner = py::none())
      : state_(load_program(
            std::move(loader),
            program_verification,
            data_path,
            std::move(owner))),
        event_tracer_(std::move(tracer)),
        debug_buffer_size_(debug_buffer_size) {
    // Figure out the size of each non_const layer we need to support every
    // method in the program. Map will be easier to use than a list because we
    // dont know how many non_const arenas there will be
    std::map<size_t, int64_t> non_const_buffer_sizes;
    for (size_t i = 0; i < state_->program_->num_methods(); ++i) {
      auto name = state_->program_->get_method_name(i).get();
      auto method_meta = state_->program_->method_meta(name).get();
      // A device-planned method gets its own arenas in load_method and never
      // reads these, so letting its sizes in would only grow the host arenas
      // the other methods share.
      if (has_device_buffers(method_meta)) {
        continue;
      }
      for (size_t j = 0; j < method_meta.num_memory_planned_buffers(); ++j) {
        auto size = method_meta.memory_planned_buffer_size(j);
        THROW_IF_ERROR(
            size.error(), "Failed to get size of planned buffer %zu", j);
        int64_t buffer_size = size.get();
        if (non_const_buffer_sizes.find(j) == non_const_buffer_sizes.end()) {
          non_const_buffer_sizes.insert({j, buffer_size});
        } else {
          non_const_buffer_sizes[j] =
              std::max(non_const_buffer_sizes[j], buffer_size);
        }
      }
    }

    // Allocate the shared host arenas.
    std::vector<int64_t> planned_sizes;
    planned_sizes.reserve(non_const_buffer_sizes.size());
    for (const auto& entry : non_const_buffer_sizes) {
      planned_sizes.push_back(entry.second);
    }

    memory_ = std::make_shared<ProgramMemory>(std::move(planned_sizes));
    if (event_tracer_ && debug_buffer_size > 0) {
      // If a debug buffer was requested for the ETDump, allocate it and make
      // sure its lifetime is as long as the event_tracer.
      debug_buffer_ = std::make_unique<uint8_t[]>(debug_buffer_size);
      event_tracer_->set_debug_buffer(get_etdump_debug_buffer());
      event_tracer_->set_event_tracer_debug_level(
          EventTracerDebugLogLevel::kIntermediateOutputs);
    }
  }

  static std::unique_ptr<PyProgram> load_from_buffer(
      const py::bytes& buffer,
      bool enable_etdump,
      size_t debug_buffer_size,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt) {
    std::unique_ptr<DataLoader> loader = loader_from_buffer(
        buffer.cast<std::string_view>().data(), py::len(buffer));
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification,
        data_path,
        buffer);
  }

  static std::unique_ptr<PyProgram> load_from_file(
      const std::string& path,
      bool enable_etdump,
      size_t debug_buffer_size,
      Program::Verification program_verification =
          Program::Verification::Minimal,
      std::optional<const std::string> data_path = std::nullopt) {
    std::unique_ptr<DataLoader> loader = loader_from_file(path);
    return std::make_unique<PyProgram>(
        std::move(loader),
        enable_etdump ? std::make_unique<torch::executor::ETDumpGen>()
                      : nullptr,
        debug_buffer_size,
        program_verification,
        data_path);
  }

  PyProgram(const PyProgram&) = delete;
  PyProgram& operator=(const PyProgram&) = delete;
  PyProgram(PyProgram&&) = default;
  PyProgram& operator=(PyProgram&&) = default;

  size_t num_methods() const {
    return state_->program_->num_methods();
  }

  std::string get_method_name(size_t method_index) const {
    Result<const char*> res = state_->program_->get_method_name(method_index);
    THROW_IF_ERROR(
        res.error(),
        "Failed get method name, error: 0x:%" PRIx32,
        static_cast<uint32_t>(res.error()));
    return std::string(res.get());
  }

  std::unique_ptr<PyMethod> load_method(const std::string& method_name) {
    Result<MethodMeta> meta =
        state_->program_->method_meta(method_name.c_str());
    THROW_IF_ERROR(
        meta.error(),
        "Failed to get method meta for method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(meta.error()));
    // Device memory is claimed here rather than at program load so that one
    // accelerator method cannot make the rest of the program unloadable. A
    // host-only method keeps sharing the program-wide arenas, so its planned
    // memory is not isolated from the other host-only methods of this program.
    auto method_memory = make_method_memory(meta.get());
    auto memory = method_memory ? std::move(method_memory) : memory_;
    Result<Method> res = state_->program_->load_method(
        method_name.c_str(),
        memory->mem_manager(),
        event_tracer_.get(),
        state_->data_map_.get());
    THROW_IF_ERROR(
        res.error(),
        "Failed to load method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return std::make_unique<PyMethod>(
        std::move(memory),
        state_,
        std::make_unique<Method>(std::move(res.get())));
  }

  Span<uint8_t> get_etdump_debug_buffer() {
    return Span<uint8_t>(debug_buffer_.get(), debug_buffer_size_);
  }

  std::unique_ptr<PyMethodMeta> method_meta(const std::string& method_name) {
    Result<torch::executor::MethodMeta> res =
        state_->program_->method_meta(method_name.c_str());
    THROW_IF_ERROR(
        res.error(),
        "Failed to get method meta for method %s, error: 0x:%" PRIx32,
        method_name.c_str(),
        static_cast<uint32_t>(res.error()));
    return std::make_unique<PyMethodMeta>(state_, std::move(res.get()));
  }

  bool has_etdump() {
    return static_cast<bool>(event_tracer_);
  }

  void write_etdump_result_to_file(
      const std::string& path,
      const py::object& debug_buffer_path) {
    if (!has_etdump()) {
      throw std::runtime_error("No etdump found");
    }
    auto& etdump = *event_tracer_;
    etdump_result result = etdump.get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      write_data_to_file(path, result.buf, result.size);
      free(result.buf);
      if (debug_buffer_size_ > 0 &&
          py::isinstance<py::str>(debug_buffer_path)) {
        // Also write out the debug buffer to a separate file if requested.
        std::string debug_buffer_path_str =
            py::cast<std::string>(debug_buffer_path);
        const auto debug_buffer = get_etdump_debug_buffer();
        write_data_to_file(
            debug_buffer_path_str, debug_buffer.data(), debug_buffer.size());
      }
    } else {
      ET_LOG(
          Info,
          "No etdump data found, try rebuilding with "
          "the CMake option EXECUTORCH_ENABLE_EVENT_TRACER set to ON or with "
          "buck run --config executorch.event_tracer_enabled=true");
    }
  }

 private:
  std::shared_ptr<ProgramMemory> memory_;
  std::shared_ptr<ProgramState> state_;
  std::unique_ptr<ETDumpGen> event_tracer_;
  std::unique_ptr<uint8_t[]> debug_buffer_;
  size_t debug_buffer_size_;
};

void create_profile_block(const std::string& name) {
  EXECUTORCH_PROFILE_CREATE_BLOCK(name.c_str());
}

py::list get_operator_names() {
  Span<const Kernel> kernels = get_registered_kernels();
  py::list res;
  for (const Kernel& k : kernels) {
    if (k.name_ != nullptr) {
      res.append(py::cast(k.name_));
    }
  }
  return res;
}

py::list get_registered_backend_names() {
  size_t n_of_registered_backends = get_num_registered_backends();
  py::list res;
  for (size_t i = 0; i < n_of_registered_backends; i++) {
    auto backend_name_res = get_backend_name(i);
    THROW_IF_ERROR(backend_name_res.error(), "Failed to get backend name");
    auto backend_name = backend_name_res.get();
    res.append(backend_name);
  }
  return res;
}

py::bool_ is_available(const std::string& backend_name) {
  BackendInterface* backend = get_backend_class(backend_name.c_str());
  if (backend == nullptr) {
    return false;
  }
  return backend->is_available();
}

} // namespace

PYBIND11_MODULE(EXECUTORCH_PYTHON_MODULE_NAME, m) {
  // Redirects cout and cerr for function calls this guards to the python env.
  auto call_guard = py::
      call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>();

  // Bind the verification enum to python.
  py::enum_<Program::Verification>(m, "Verification")
      .value("Minimal", Program::Verification::Minimal)
      .value("InternalConsistency", Program::Verification::InternalConsistency);

  m.def(
      "_load_for_executorch",
      PyModule::load_from_file,
      py::arg("program_path"),
      py::arg("data_path") = std::nullopt,
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") =
          Program::Verification::InternalConsistency,
      call_guard);
  m.def(
      "_load_for_executorch_from_buffer",
      &PyModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("data_map_buffer") = std::nullopt,
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") =
          Program::Verification::InternalConsistency,
      call_guard);
  m.def(
      "_load_for_executorch_from_bundled_program",
      py::overload_cast<
          PyBundledModule&,
          std::optional<const py::bytes>,
          bool,
          size_t>(&PyModule::load_from_bundled_program),
      py::arg("ptr"),
      py::arg("data_map_buffer") = std::nullopt,
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      call_guard);
  m.def(
      "_load_for_executorch_from_bundled_program",
      py::overload_cast<PyBundledModule&, const std::string&, bool, size_t>(
          &PyModule::load_from_bundled_program),
      py::arg("ptr"),
      py::arg("data_path"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      call_guard);
  m.def(
      "_load_bundled_program_from_buffer",
      &PyBundledModule::load_from_buffer,
      py::arg("buffer"),
      py::arg("non_const_pool_size") = kDEFAULT_BUNDLED_INPUT_POOL_SIZE,
      call_guard);

  // Import the PyDataLoader type from the shared module.
  // This ensures the type is registered once and shared across all modules.
  py::module_::import("executorch.extension.pybindings.data_loader");

  m.def(
      "_load_for_executorch_from_data_loader",
      &PyModule::load_from_data_loader,
      py::arg("loader"),
      py::arg("data_path") = py::none(),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      call_guard);

  m.def(
      "_dump_profile_results",
      []() {
        prof_result_t prof_result;
        EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
        return py::bytes(
            reinterpret_cast<const char*>(prof_result.prof_data),
            prof_result.num_bytes);
      },
      call_guard);
  m.def(
      "_get_registered_backend_names",
      &get_registered_backend_names,
      call_guard);
  m.def("_get_operator_names", &get_operator_names);
  m.def("_is_available", &is_available, py::arg("backend_name"), call_guard);
  m.def("_create_profile_block", &create_profile_block, call_guard);
  m.def(
      "_reset_profile_results",
      []() { EXECUTORCH_RESET_PROFILE_RESULTS(); },
      call_guard);
  m.def(
      "_unsafe_reset_threadpool",
      [](int num_threads) {
        executorch::extension::threadpool::get_threadpool()
            ->_unsafe_reset_threadpool(num_threads);
      },
      py::arg("num_threads"),
      call_guard);
  m.def(
      "_threadpool_get_thread_count",
      []() {
        return ::executorch::extension::threadpool::get_threadpool()
            ->get_thread_count();
      },
      call_guard);

  py::class_<PyModule>(m, "ExecuTorchModule")
      .def(
          "plan_execute",
          &PyModule::plan_execute,
          py::arg("method_name"),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "method_meta",
          &PyModule::method_meta,
          py::arg("method_name"),
          call_guard)
      .def("method_names", &PyModule::method_names, call_guard)
      .def(
          "run_method",
          &PyModule::run_method,
          py::arg("method_name"),
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "forward",
          &PyModule::forward,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def("has_etdump", &PyModule::has_etdump, call_guard)
      .def(
          "write_etdump_result_to_file",
          &PyModule::write_etdump_result_to_file,
          py::arg("path"),
          py::arg("debug_buffer_path") = py::none(),
          call_guard)
      .def(
          "__call__",
          &PyModule::forward,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard);

  py::class_<PyBundledModule>(m, "BundledModule")
      .def(
          "verify_result_with_bundled_expected_output",
          &PyBundledModule::verify_result_with_bundled_expected_output,
          py::arg("method_name"),
          py::arg("testset_idx"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
          call_guard);

#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
  py::class_<PyTensorBuffer>(m, "TensorBuffer", py::buffer_protocol())
      .def_buffer(&PyTensorBuffer::buffer_info)
      .def_property_readonly("shape", &PyTensorBuffer::shape)
      .def_property_readonly("dtype", &PyTensorBuffer::dtype)
      .def_property_readonly("nbytes", &PyTensorBuffer::nbytes)
      .def(
          "__dlpack__",
          [](py::object self, py::object stream) {
            return py::cast<PyTensorBuffer*>(self)->dlpack(self, stream);
          },
          py::arg("stream") = py::none())
      .def("__dlpack_device__", &PyTensorBuffer::dlpack_device)
      .def("__repr__", &PyTensorBuffer::repr, call_guard);
#endif // EXECUTORCH_PYBINDINGS_WITHOUT_TORCH

  py::class_<PyTensorInfo>(m, "TensorInfo")
      .def("sizes", &PyTensorInfo::sizes, call_guard)
      .def("dtype", &PyTensorInfo::dtype, call_guard)
      .def("is_memory_planned", &PyTensorInfo::is_memory_planned, call_guard)
      .def("nbytes", &PyTensorInfo::nbytes, call_guard)
      .def("__repr__", &PyTensorInfo::repr, call_guard);
  py::class_<PyMethodMeta>(m, "MethodMeta")
      .def("name", &PyMethodMeta::name, call_guard)
      .def("num_inputs", &PyMethodMeta::num_inputs, call_guard)
      .def("num_outputs", &PyMethodMeta::num_outputs, call_guard)
      .def("num_attributes", &PyMethodMeta::num_attributes, call_guard)
      .def(
          "input_tensor_meta",
          &PyMethodMeta::input_tensor_meta,
          py::arg("index"),
          call_guard)
      .def(
          "output_tensor_meta",
          &PyMethodMeta::output_tensor_meta,
          py::arg("index"),
          call_guard)
      .def(
          "attribute_tensor_meta",
          &PyMethodMeta::attribute_tensor_meta,
          py::arg("index"),
          call_guard)
      .def("__repr__", &PyMethodMeta::repr, call_guard);

  m.def(
      "_load_program",
      &PyProgram::load_from_file,
      py::arg("path"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") = Program::Verification::Minimal,
      py::arg("data_path") = std::nullopt,
      call_guard);
  m.def(
      "_load_program_from_buffer",
      &PyProgram::load_from_buffer,
      py::arg("buffer"),
      py::arg("enable_etdump") = false,
      py::arg("debug_buffer_size") = 0,
      py::arg("program_verification") = Program::Verification::Minimal,
      py::arg("data_path") = std::nullopt,
      call_guard);
#ifdef EXECUTORCH_PYBINDINGS_WITHOUT_TORCH
  m.attr("_links_torch") = false;
#else
  m.attr("_links_torch") = true;
#endif

  py::class_<PyProgram>(m, "ExecuTorchProgram")
      .def("num_methods", &PyProgram::num_methods, call_guard)
      .def(
          "get_method_name",
          &PyProgram::get_method_name,
          py::arg("method_index"),
          call_guard)
      .def(
          "load_method",
          &PyProgram::load_method,
          py::arg("method_name"),
          call_guard)
      .def(
          "method_meta",
          &PyProgram::method_meta,
          py::arg("method_name"),
          call_guard)
      .def("has_etdump", &PyProgram::has_etdump, call_guard)
      .def(
          "write_etdump_result_to_file",
          &PyProgram::write_etdump_result_to_file,
          py::arg("path"),
          py::arg("debug_buffer_path") = py::none(),
          call_guard);
  py::class_<PyMethod>(m, "ExecuTorchMethod")
      .def("set_inputs", &PyMethod::set_inputs, py::arg("inputs"), call_guard)
      .def("execute", &PyMethod::execute, call_guard)
      .def(
          "get_outputs",
          &PyMethod::get_outputs,
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "call",
          &PyMethod::call,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "__call__",
          &PyMethod::call,
          py::arg("inputs") = py::list(),
          py::arg("clone_outputs") = true,
          call_guard)
      .def(
          "get_attribute",
          &PyMethod::get_attribute,
          py::arg("name"),
          call_guard)
      .def("method_meta", &PyMethod::method_meta, call_guard);
}

namespace {

// Our logs work by writing to stderr. By default this is done through fprintf
// (as defined in posix.cpp) which then does not show up in python environments.
// Here we override the pal to use std::cerr which can be properly redirected by
// scoped_estream_redirect.
void emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  std::cerr << "[" << filename << ":" << line << "] " << message << std::endl;
}

runtime::PalImpl build_pal() {
  return runtime::PalImpl::create(emit_log_message, __FILE__);
}

// Update PAL to redirect logs.
ET_UNUSED bool registration_result = runtime::register_pal(build_pal());

} // namespace

} // namespace pybindings
} // namespace extension
} // namespace executorch
