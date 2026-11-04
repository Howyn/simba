"""Build the baseline CUDA extension from this repository."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

source = Path(__file__).resolve().parent / 'csrc' / 'selective_scan'
names = ['selective_scan.cpp', 'selective_scan_fwd_fp32.cu',
         'selective_scan_fwd_fp16.cu', 'selective_scan_fwd_bf16.cu',
         'selective_scan_bwd_fp32_real.cu', 'selective_scan_bwd_fp32_complex.cu',
         'selective_scan_bwd_fp16_real.cu', 'selective_scan_bwd_fp16_complex.cu',
         'selective_scan_bwd_bf16_real.cu', 'selective_scan_bwd_bf16_complex.cu']
setup(name='simba-baseline-scan', version='0.1.0',
      ext_modules=[CUDAExtension('selective_scan_cuda',
          sources=[str(source / name) for name in names], include_dirs=[str(source)],
          extra_compile_args={'cxx': ['-O3', '-std=c++17'],
             'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                      '--expt-extended-lambda', '--use_fast_math', '-lineinfo']})],
      cmdclass={'build_ext': BuildExtension})
