### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'generated_kernel': {
        'source_files': {
            'h100': 'kernels/generated_kernel/generated_kernel.cu'  
        }
    }
}

kernels = ['generated_kernel']

target = 'h100'

