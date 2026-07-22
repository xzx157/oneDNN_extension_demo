#include <ATen/Functions.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ODNN_DEMO_USE_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace odnn_demo {

#ifdef ODNN_DEMO_USE_DNNL
namespace {

using Dims = dnnl::memory::dims;

Dims to_dims(at::IntArrayRef values) {
  return Dims(values.begin(), values.end());
}

Dims public_strides(const Dims& sizes, bool channels_last) {
  TORCH_CHECK(sizes.size() == 4, "expected a four-dimensional tensor");
  if (channels_last) {
    return {
        sizes[1] * sizes[2] * sizes[3], 1, sizes[3] * sizes[1], sizes[1]};
  }
  return {
      sizes[1] * sizes[2] * sizes[3],
      sizes[2] * sizes[3],
      sizes[3],
      1};
}

Dims convolution_output_size(
    const Dims& input,
    const Dims& weight,
    const Dims& stride,
    const Dims& padding,
    const Dims& dilation) {
  return {
      input[0],
      weight[0],
      (input[2] + 2 * padding[0] - dilation[0] * (weight[2] - 1) - 1) /
              stride[0] +
          1,
      (input[3] + 2 * padding[1] - dilation[1] * (weight[3] - 1) - 1) /
              stride[1] +
          1};
}

class NativeDnnlConvolution {
 public:
  NativeDnnlConvolution(
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation,
      int64_t groups,
      const std::vector<int64_t>& input_size,
      bool channels_last)
      : engine_(dnnl::engine::kind::cpu, 0),
        stream_(engine_),
        input_size_(input_size.begin(), input_size.end()),
        channels_last_(channels_last) {
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "native oneDNN supports FP32");
    TORCH_CHECK(weight.device().is_cpu(), "native oneDNN weight must be on CPU");
    TORCH_CHECK(input_size_.size() == 4, "sample_input must be four-dimensional");

    const auto weight_dims = to_dims(weight.sizes());
    const auto stride_dims = Dims(stride.begin(), stride.end());
    const auto padding_dims = Dims(padding.begin(), padding.end());
    const auto dilation_dims = Dims(dilation.begin(), dilation.end());
    const auto output_dims = convolution_output_size(
        input_size_, weight_dims, stride_dims, padding_dims, dilation_dims);

    src_strides_ = public_strides(input_size_, channels_last_);
    dst_strides_ = public_strides(output_dims, channels_last_);
    src_desc_ = dnnl::memory::desc(
        input_size_, dnnl::memory::data_type::f32, src_strides_);
    dst_desc_ = dnnl::memory::desc(
        output_dims, dnnl::memory::data_type::f32, dst_strides_);

    Dims logical_weight_dims;
    Dims logical_weight_strides;
    if (groups == 1) {
      logical_weight_dims = weight_dims;
      logical_weight_strides = to_dims(weight.strides());
    } else {
      TORCH_CHECK(weight_dims[0] % groups == 0, "invalid grouped weight");
      logical_weight_dims = {
          groups,
          weight_dims[0] / groups,
          weight_dims[1],
          weight_dims[2],
          weight_dims[3]};
      const auto weight_strides = to_dims(weight.strides());
      logical_weight_strides = {
          (weight_dims[0] / groups) * weight_strides[0],
          weight_strides[0],
          weight_strides[1],
          weight_strides[2],
          weight_strides[3]};
    }

    const auto user_weight_desc = dnnl::memory::desc(
        logical_weight_dims,
        dnnl::memory::data_type::f32,
        logical_weight_strides);
    const auto requested_weight_desc = dnnl::memory::desc(
        logical_weight_dims,
        dnnl::memory::data_type::f32,
        dnnl::memory::format_tag::any);
    const auto bias_desc = bias.has_value()
        ? dnnl::memory::desc(
              {weight_dims[0]},
              dnnl::memory::data_type::f32,
              dnnl::memory::format_tag::x)
        : dnnl::memory::desc();
    Dims dnnl_dilation = {dilation_dims[0] - 1, dilation_dims[1] - 1};

    if (bias.has_value()) {
      primitive_desc_ = dnnl::convolution_forward::primitive_desc(
          engine_,
          dnnl::prop_kind::forward_inference,
          dnnl::algorithm::convolution_direct,
          src_desc_,
          requested_weight_desc,
          bias_desc,
          dst_desc_,
          stride_dims,
          dnnl_dilation,
          padding_dims,
          padding_dims);
    } else {
      primitive_desc_ = dnnl::convolution_forward::primitive_desc(
          engine_,
          dnnl::prop_kind::forward_inference,
          dnnl::algorithm::convolution_direct,
          src_desc_,
          requested_weight_desc,
          dst_desc_,
          stride_dims,
          dnnl_dilation,
          padding_dims,
          padding_dims);
    }
    primitive_ = dnnl::convolution_forward(primitive_desc_);
    packed_weight_ = dnnl::memory(primitive_desc_.weights_desc(), engine_);

    auto user_weight = dnnl::memory(
        user_weight_desc, engine_, const_cast<void*>(weight.const_data_ptr()));
    dnnl::reorder(user_weight, packed_weight_)
        .execute(stream_, user_weight, packed_weight_);
    stream_.wait();

    src_memory_ = dnnl::memory(src_desc_, engine_, DNNL_MEMORY_NONE);
    dst_memory_ = dnnl::memory(dst_desc_, engine_, DNNL_MEMORY_NONE);
    arguments_ = {
        {DNNL_ARG_SRC, src_memory_},
        {DNNL_ARG_WEIGHTS, packed_weight_},
        {DNNL_ARG_DST, dst_memory_}};
    if (bias.has_value()) {
      bias_memory_ = dnnl::memory(
          primitive_desc_.bias_desc(),
          engine_,
          const_cast<void*>(bias.value().const_data_ptr()));
      arguments_.emplace(DNNL_ARG_BIAS, bias_memory_);
    }
  }

  bool can_run(const at::Tensor& input) const {
    return input.device().is_cpu() && input.scalar_type() == at::kFloat &&
        input.layout() == c10::kStrided && to_dims(input.sizes()) == input_size_ &&
        to_dims(input.strides()) == src_strides_;
  }

  at::Tensor run(const at::Tensor& input) const {
    auto output = at::empty(
        primitive_desc_.dst_desc().get_dims(),
        input.options().memory_format(
            channels_last_ ? at::MemoryFormat::ChannelsLast
                           : at::MemoryFormat::Contiguous));
    std::lock_guard<std::mutex> guard(execution_mutex_);
    src_memory_.set_data_handle(const_cast<void*>(input.const_data_ptr()));
    dst_memory_.set_data_handle(output.data_ptr());
    primitive_.execute(stream_, arguments_);
    stream_.wait();
    return output;
  }

  int64_t packed_weight_bytes() const {
    return static_cast<int64_t>(primitive_desc_.weights_desc().get_size());
  }

 private:
  dnnl::engine engine_;
  mutable dnnl::stream stream_;
  Dims input_size_;
  Dims src_strides_;
  Dims dst_strides_;
  bool channels_last_;
  dnnl::memory::desc src_desc_;
  dnnl::memory::desc dst_desc_;
  dnnl::convolution_forward::primitive_desc primitive_desc_;
  dnnl::convolution_forward primitive_;
  dnnl::memory packed_weight_;
  mutable dnnl::memory src_memory_;
  mutable dnnl::memory dst_memory_;
  dnnl::memory bias_memory_;
  mutable std::unordered_map<int, dnnl::memory> arguments_;
  mutable std::mutex execution_mutex_;
};

} // namespace
#endif

class ConvolutionOpContext : public torch::CustomClassHolder {
 public:
  ConvolutionOpContext(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      int64_t groups,
      std::vector<int64_t> input_size,
      bool channels_last)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        input_size_(std::move(input_size)),
        channels_last_(channels_last) {
    TORCH_CHECK(
        weight_.layout() == c10::kStrided,
        "ConvolutionOpContext expects a strided weight");
#ifdef ODNN_DEMO_USE_DNNL
    if (weight_.scalar_type() == at::kFloat && input_size_.size() == 4) {
      try {
        native_context_ = std::make_shared<NativeDnnlConvolution>(
            weight_,
            bias_,
            stride_,
            padding_,
            dilation_,
            groups_,
            input_size_,
            channels_last_);
      } catch (const dnnl::error&) {
        native_context_.reset();
      }
    }
#endif
  }

  at::Tensor run(const at::Tensor& input) const {
    TORCH_CHECK(
        input.layout() == c10::kStrided,
        "convolution input must be strided");
#ifdef ODNN_DEMO_USE_DNNL
    if (native_context_ && native_context_->can_run(input)) {
      native_runs_.fetch_add(1, std::memory_order_relaxed);
      return native_context_->run(input);
    }
#endif
    fallback_runs_.fetch_add(1, std::memory_order_relaxed);
    return at::conv2d(
        input,
        weight_,
        bias_,
        stride_,
        padding_,
        dilation_,
        groups_);
  }

  at::Tensor get_packed_weight() const {
    return weight_;
  }

  std::vector<int64_t> get_input_size() const {
    return input_size_;
  }

  bool uses_native_dnnl() const {
#ifdef ODNN_DEMO_USE_DNNL
    return static_cast<bool>(native_context_);
#else
    return false;
#endif
  }

  int64_t packed_weight_bytes() const {
#ifdef ODNN_DEMO_USE_DNNL
    return native_context_ ? native_context_->packed_weight_bytes() : 0;
#else
    return 0;
#endif
  }

  int64_t native_runs() const {
    return native_runs_.load(std::memory_order_relaxed);
  }

  int64_t fallback_runs() const {
    return fallback_runs_.load(std::memory_order_relaxed);
  }

 private:
  at::Tensor weight_;
  c10::optional<at::Tensor> bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::vector<int64_t> input_size_;
  bool channels_last_;
  mutable std::atomic<int64_t> native_runs_{0};
  mutable std::atomic<int64_t> fallback_runs_{0};
#ifdef ODNN_DEMO_USE_DNNL
  std::shared_ptr<NativeDnnlConvolution> native_context_;
#endif
};

class LinearOpContext : public torch::CustomClassHolder {
 public:
  LinearOpContext(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      std::vector<int64_t> input_size)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        input_size_(std::move(input_size)) {
    TORCH_CHECK(
        weight_.layout() == c10::kStrided,
        "LinearOpContext expects a strided weight");
  }

  at::Tensor run(const at::Tensor& input) const {
    TORCH_CHECK(input.layout() == c10::kStrided, "linear input must be strided");
    return at::linear(input, weight_, bias_);
  }

  at::Tensor get_packed_weight() const {
    return weight_;
  }

  std::vector<int64_t> get_input_size() const {
    return input_size_;
  }

 private:
  at::Tensor weight_;
  c10::optional<at::Tensor> bias_;
  std::vector<int64_t> input_size_;
};

} // namespace odnn_demo

TORCH_LIBRARY(odnn_prepack, m) {
  m.class_<odnn_demo::ConvolutionOpContext>("ConvolutionOpContext")
      .def(torch::init<
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>,
           std::vector<int64_t>,
           int64_t,
           std::vector<int64_t>,
           bool>())
      .def("run", &odnn_demo::ConvolutionOpContext::run)
      .def(
          "get_packed_weight",
          &odnn_demo::ConvolutionOpContext::get_packed_weight)
      .def("get_input_size", &odnn_demo::ConvolutionOpContext::get_input_size)
      .def("uses_native_dnnl", &odnn_demo::ConvolutionOpContext::uses_native_dnnl)
      .def(
          "packed_weight_bytes",
          &odnn_demo::ConvolutionOpContext::packed_weight_bytes)
      .def("native_runs", &odnn_demo::ConvolutionOpContext::native_runs)
      .def("fallback_runs", &odnn_demo::ConvolutionOpContext::fallback_runs);

  m.class_<odnn_demo::LinearOpContext>("LinearOpContext")
      .def(torch::init<
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>>())
      .def("run", &odnn_demo::LinearOpContext::run)
      .def(
          "get_packed_weight",
          &odnn_demo::LinearOpContext::get_packed_weight)
      .def("get_input_size", &odnn_demo::LinearOpContext::get_input_size);
}
