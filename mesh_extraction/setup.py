from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

mise_module = Extension("mise", sources=["/home/mv2mp/mesh_extraction/src/mise.pyx"])
setup(
    name="mesh-extraction",
    ext_modules=cythonize([mise_module]),
    version="2024.2",
)
