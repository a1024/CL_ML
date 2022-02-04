Machine Learning with OpenCL

A framework to test and hopefully train convolutional networks
on non-CUDA devices.


BUILD:
- Download the OpenCL headers from:
https://github.com/KhronosGroup/OpenCL-Headers

Building on Linux:
- Rename 'Makefile-mingw64' to 'Makefile'
- Set the path to the OpenCL headers in the Makefile
- Open Mingw64 shell and type 'make'.
  should work on MSYS2 as well

Building with Ms Visual C++:
- Add all source files to a 64-bit console project


USAGE:
- Place 'cl_kernels.h' in same directory as ml64
- Extract the ResNet18 weights as lossless precision text
  with the script 'extract_weights.py'.
- Run ml64 for the first time as:
	ml64 weights path/to/weights.txt
- Optional: place the generated .bin files with ml64, to avoid
  passing the path to them
- Run ml64 subsequently with the target image:
	ml64 path/to/targetimage [path/to/weights.bin]


NOTE:
This project is still a work in progress.
