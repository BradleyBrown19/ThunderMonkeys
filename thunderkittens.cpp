#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>


#ifdef TK_COMPILE_GENERATED_KERNEL
extern void generated_kernel(
    int has_residual,
    float dropout_p,
    torch::Tensor x,
    torch::Tensor residual, 
    torch::Tensor norm_weight, torch::Tensor norm_bias, 
    torch::Tensor o, torch::Tensor out_resid
);
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Kernels"; // optional module docstring

#ifdef TK_COMPILE_GENERATED_KERNEL
    m.def("generated_kernel", generated_kernel, "Generated kernel.");
#endif

}
 
