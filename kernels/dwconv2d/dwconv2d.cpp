#include <torch/extension.h>

at::Tensor Dwconv2dLauncherFP32(const at::Tensor input, const at::Tensor weight, const int padding_h, const int padding_w);
at::Tensor Dwconv2dBiasLauncherFP32(const at::Tensor input, const at::Tensor weight, const at::Tensor bias, const int padding_h, const int padding_w);
at::Tensor Dwconv2dLauncherFP32Wgrad(const at::Tensor input, const at::Tensor ograd, const int padding_w, const int padding_h);
at::Tensor Dwconv2dLauncherFP32Bgrad(const at::Tensor ograd);
at::Tensor Dwconv2dLauncherFP32Dgrad(const at::Tensor Ograd, const at::Tensor weight_inv, const int padding_h, const int padding_w);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, "must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor dwconv2d_fp32(const at::Tensor input, const at::Tensor weight, const int padding_h, const int padding_w)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    at::DeviceGuard guard(input.device());
    return Dwconv2dLauncherFP32(input, weight, padding_h, padding_w);
}

at::Tensor dwconv2dbias_fp32(const at::Tensor input, const at::Tensor weight, const at::Tensor bias, const int padding_h, const int padding_w)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    at::DeviceGuard guard(input.device());
    return Dwconv2dBiasLauncherFP32(input, weight, bias, padding_h, padding_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dwconv2d_fp32", &dwconv2d_fp32, "dwconv2d_fp32");
    m.def("dwconv2dbias_fp32", &dwconv2dbias_fp32, "dwconv2dbias_fp32");

}

