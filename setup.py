"""The C extension; everything else is in pyproject.toml."""
import platform

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """numpy >= 2.5 already requires x86-64-v2 (SSE4.2) on x86-64, so the NA
    scans may use it: SSE2 has no 64-bit compare, and they vectorise badly
    without one.  GCC and Clang only; MSVC and other CPUs keep their default."""

    def build_extensions(self):
        x86_64 = platform.machine().lower() in ("x86_64", "amd64")
        if x86_64 and self.compiler.compiler_type != "msvc":
            for ext in self.extensions:
                ext.extra_compile_args += ["-march=x86-64-v2", "-mtune=generic"]
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "_nulldtype",
            sources=["src/nulldtype.c"],
            include_dirs=[np.get_include()],
        ),
    ],
    cmdclass={"build_ext": BuildExt},
)
