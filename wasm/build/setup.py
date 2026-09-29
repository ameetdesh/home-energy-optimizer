from setuptools import setup, Extension
from Cython.Build import cythonize

# -Os rather than -O3: the wheel is base64'd into an HTML file, so binary size
# is the budget that matters more than peak throughput, and these are tight
# scalar loops that -Os compiles about as well.
extensions = [
    Extension("dp_kernels", ["dp_kernels.pyx"],
              extra_compile_args=["-Os", "-g0"],
              extra_link_args=["-Os", "-g0"]),
]

setup(
    name="dp_kernels",
    version="0.1.0",
    ext_modules=cythonize(extensions, compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
)
