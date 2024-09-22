### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'generated_kernel': {
        'source_files': {
            'h100': ['kernels/generated_kernel/generated_kernel.cu']  
        }
    },
    'fused_layernorm': {
        'source_files': {
            'h100': [
                'kernels/fused_layernorm/layer_norm.cu',
            ]
        }
    },
}

kernels = ["fused_layernorm"]

target = 'h100'

