#pragma once
#ifndef UBENCH_H
#define UBENCH_H
#include"util.h"

	#define ENABLE_OPENCL

typedef struct CPUInfoStruct
{
	char vendor[32], brand[64];
	char
		mmx,
		sse,
		sse2,
		sse3,
		ssse3,
		sse4_1,
		sse4_2,
		fma,
		aes,
		sha,
		avx,
		avx2,
		avx512F,
		avx512PF,
		avx512ER,
		avx512CD,
		f16c,
		rdrand,
		rdseed;
} CPUInfo;
void get_cpu_info(CPUInfo *info);

int test_scalar_fp64	(unsigned char *image, int iw, int ih);
int test_scalar_fp32	(unsigned char *image, int iw, int ih);
int test_scalar_f16p16	(unsigned char *image, int iw, int ih);

int test_sse_fp32	(unsigned char *image, int iw, int ih);
int test_sse2_fp64	(unsigned char *image, int iw, int ih);

int test_avx_fp32	(unsigned char *image, int iw, int ih);
int test_avx_fp64	(unsigned char *image, int iw, int ih);

int test_avx512_fp32	(unsigned char *image, int iw, int ih);
int test_avx512_fp64	(unsigned char *image, int iw, int ih);

#ifdef ENABLE_OPENCL
int test_cl(const char *programname, unsigned char *image, int iw, int ih);//cwd is changed by command args
void print_clinfo();
#endif

#endif//UBENCH_H
