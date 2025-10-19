from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        'nptest',
        ['nptest.cpp'],
        include_dirs=['BLAS/include', pybind11.get_include()],
        library_dirs=['BLAS/lib'],
        libraries=['libopenblas'],  # если установлен OpenBLAS
        extra_compile_args=[
                    '/O2',          # максимум разумной оптимизации
                    '/GL',          # link-time code generation
                    '/fp:fast',     # быстрая математика
                    '/arch:AVX2',   # векторизация под твой CPU (если поддерживается)
                    '/std:c++17'
                             ],
    ),
]

setup(
    name='test',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)