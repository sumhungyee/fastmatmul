from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "matmul",
        sorted(glob("src/*.cpp")), 
    ),
]

setup(
    name='matmul',
    version='0.0.1',
    install_requires=[
        'requests',
        'importlib-metadata; python_version<"3.10"',
    ], ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
    )