from distutils.core import setup, Extension
import numpy as np

setup(
	name="pyentropy",
	version="1.0",
	ext_modules=[Extension("pyentropy", sources=["pyentropy.c"], include_dirs=[
		np.get_include()
	])]
)
