from setuptools import setup
import torch.utils.cpp_extension
# Monkey patch to bypass CUDA version check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polynomial_cuda',
    ext_modules=[
        CUDAExtension('polynomial_cuda', [
            'polynomial_cuda.cu',
        ],
        extra_compile_args={
            'cxx': ['/Zc:preprocessor'],
            'nvcc': ['-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING', '-Xcompiler', '/Zc:preprocessor']
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })