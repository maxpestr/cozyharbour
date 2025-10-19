from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "core",
        ["core.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="core",
    version="0.0.1",
    author="You",
    description="Simple particle simulation core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
