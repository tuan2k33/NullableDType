"""The C extension; everything else is in pyproject.toml."""
import numpy as np
from setuptools import Extension, setup

setup(ext_modules=[
    Extension(
        "_nulldtype",
        sources=["src/nulldtype.c"],
        include_dirs=[np.get_include()],
    ),
])
