#include <ATen/Functions.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <utility>
#include <vector>

namespace odnn_demo {

class ConvolutionOpContext : public torch::CustomClassHolder {
 public:
  ConvolutionOpContext(
      at::Tensor packed_weight,
      c10::optional<at::Tensor> bias,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      int64_t groups,
      std::vector<int64_t> input_size)
      : packed_weight_(std::move(packed_weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        input_size_(std::move(input_size)) {
    TORCH_CHECK(
        packed_weight_.is_mkldnn(),
        "ConvolutionOpContext expects an MKLDNN packed weight");
  }

  at::Tensor run(const at::Tensor& input) const {
    TORCH_CHECK(input.is_mkldnn(), "convolution input must be MKLDNN");
    return at::conv2d(
        input,
        packed_weight_,
        bias_,
        stride_,
        padding_,
        dilation_,
        groups_);
  }

  at::Tensor get_packed_weight() const {
    return packed_weight_;
  }

  std::vector<int64_t> get_input_size() const {
    return input_size_;
  }

 private:
  at::Tensor packed_weight_;
  c10::optional<at::Tensor> bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::vector<int64_t> input_size_;
};

class LinearOpContext : public torch::CustomClassHolder {
 public:
  LinearOpContext(
      at::Tensor packed_weight,
      c10::optional<at::Tensor> bias,
      std::vector<int64_t> input_size)
      : packed_weight_(std::move(packed_weight)),
        bias_(std::move(bias)),
        input_size_(std::move(input_size)) {
    TORCH_CHECK(
        packed_weight_.is_mkldnn(),
        "LinearOpContext expects an MKLDNN packed weight");
  }

  at::Tensor run(const at::Tensor& input) const {
    TORCH_CHECK(input.is_mkldnn(), "linear input must be MKLDNN");
    return at::mkldnn_linear(input, packed_weight_, bias_);
  }

  at::Tensor get_packed_weight() const {
    return packed_weight_;
  }

  std::vector<int64_t> get_input_size() const {
    return input_size_;
  }

 private:
  at::Tensor packed_weight_;
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
           std::vector<int64_t>>())
      .def("run", &odnn_demo::ConvolutionOpContext::run)
      .def(
          "get_packed_weight",
          &odnn_demo::ConvolutionOpContext::get_packed_weight)
      .def("get_input_size", &odnn_demo::ConvolutionOpContext::get_input_size);

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
