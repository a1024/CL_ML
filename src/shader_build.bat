@echo off
REM
REM Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
REM
REM Permission is hereby granted, free of charge, to any person obtaining a
REM copy of this software and associated documentation files (the "Software"),
REM to deal in the Software without restriction, including without limitation
REM the rights to use, copy, modify, merge, publish, distribute, sublicense,
REM and/or sell copies of the Software, and to permit persons to whom the
REM Software is furnished to do so, subject to the following conditions:
REM
REM The above copyright notice and this permission notice shall be included in
REM all copies or substantial portions of the Software.
REM
REM THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
REM IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
REM FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
REM THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
REM LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
REM FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
REM DEALINGS IN THE SOFTWARE.
REM
@echo on
glslangValidator -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=16 -DC_TYPE=float16_t -DcoopmatT=fcoopmatNV -V shader_tiled.comp -o shader_tiledfp16.spv
glslangValidator -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=32 -DC_TYPE=float     -DcoopmatT=fcoopmatNV -V shader_tiled.comp -o shader_tiledfp32.spv
glslangValidator -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=16 -DC_TYPE=float16_t -DcoopmatT=fcoopmatNV -V shader_shmem.comp -o shader_shmemfp16.spv
glslangValidator -DA_BITS=16 -DA_TYPE=float16_t -DC_BITS=32 -DC_TYPE=float     -DcoopmatT=fcoopmatNV -V shader_shmem.comp -o shader_shmemfp32.spv
glslangValidator -DA_BITS=8  -DA_TYPE=uint8_t   -DC_BITS=32 -DC_TYPE=uint32_t  -DcoopmatT=ucoopmatNV -V shader_tiled.comp -o shader_tiledu8.spv
glslangValidator -DA_BITS=8  -DA_TYPE=uint8_t   -DC_BITS=32 -DC_TYPE=uint32_t  -DcoopmatT=ucoopmatNV -V shader_shmem.comp -o shader_shmemu8.spv
glslangValidator -DA_BITS=8  -DA_TYPE=int8_t    -DC_BITS=32 -DC_TYPE=int32_t   -DcoopmatT=icoopmatNV -V shader_tiled.comp -o shader_tileds8.spv
glslangValidator -DA_BITS=8  -DA_TYPE=int8_t    -DC_BITS=32 -DC_TYPE=int32_t   -DcoopmatT=icoopmatNV -V shader_shmem.comp -o shader_shmems8.spv