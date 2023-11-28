#ifndef __OPEN_CL__
#include<math.h>
#include<stdio.h>
#define __kernel
#define __global
#define __constant
#define kernel
#define global
#define constant
#define get_global_id(...)	(__VA_ARGS__)
#define get_global_size(...)	(__VA_ARGS__)
#define min(...)		(__VA_ARGS__)
#define max(...)		(__VA_ARGS__)
#define mix(...)		(__VA_ARGS__)
#define clamp(...)		(__VA_ARGS__)
#define as_float(...)		(__VA_ARGS__)
#define as_int(...)		(__VA_ARGS__)
#endif

	#define CONST __constant
//	#define CONST __global const
#define CLAMP(LO, X, HI) ((X)>(LO)?(X)<(HI)?(X):(HI):(LO))
#ifdef PREC_FIXED
typedef int DataType;
#define CVT_F2D(X) (int)((X)*(1<<PREC_FIXED))
#define CVT_D2F(X) ((float)(X)*(1.f/(1<<PREC_FIXED)))
#define CVT_I2D(X) ((X)*(1<<PREC_FIXED)/512)
#define CVT_D2I(X) (CLAMP(-(1<<PREC_FIXED), X, (1<<PREC_FIXED))*127/(1<<PREC_FIXED))
#define MUL(A, B)	((long)(A)*(B)>>PREC_FIXED)
#define ONE_PERCENT	((1<<PREC_FIXED)/100)
#define SQRT(X)		(int)(sqrt((float)(X)/(1<<PREC_FIXED))*(1<<PREC_FIXED))//TODO fixed_sqrt
#define DIV(N, D)	(int)(((long)(N)<<PREC_FIXED)/(D))
#define ZERO		0
#define ONE		(1<<PREC_FIXED)
#define MIX(A, B, X)	((A)+(((B)-(A))*(X)>>PREC_FIXED))
int ROUND(int x)
{
	int neg=x>>31&1;

	x^=-neg;//remove sign
	x+=neg;

	x+=0x8000;//sgn(x)*trunc(abs(x)+0.5)
	x&=0xFFFF0000;

	x^=-neg;//restore sign
	x+=neg;

	return x;
}
//#define	PRINT		"%08X"
#define		PRINT		"%d"
#elif defined PREC_HALF//not supported on nVidia's OpenCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half DataType;
#define CVT_F2D(X) (half)(X)
#define CVT_D2F(X) (float)(X)
#define CVT_I2D(X) ((X)*(1/127.h))
#define CVT_D2I(X) (int)(CLAMP(-1, X, 1)*512.h)
#define MUL(A, B)	((A)*(B))
#define ONE_PERCENT	0.01h
#define SQRT		sqrt
#define DIV(N, D)	(N/D)
#define ZERO		0.h
#define ONE		1.h
#define MIX		mix
#define ROUND		round
#define PRINT		"%04h"
#else
typedef float DataType;
#define CVT_F2D(X) (X)
#define CVT_D2F(X) (X)
#define CVT_I2D(X) ((X)*(1/512.f))
#define CVT_D2I(X) (int)((CLAMP(-1, X, 1)+1)*127.f)
#define MUL(A, B)	((A)*(B))
#define ONE_PERCENT	0.01f
#define SQRT		sqrt
#define DIV(N, D)	(N/D)
#define ZERO		0.f
#define ONE		1.f
#define MIX		mix
#define ROUND		round
#define PRINT		"%g"
#endif

__kernel void buf2tensor(__global const float *src, __global DataType *dst)
{
	int idx=get_global_id(0);
	dst[idx]=CVT_F2D(src[idx]);
}
__kernel void image2tensor(__global const int *src, __global DataType *dst)
{
	int idx=get_global_id(0);
	int val=src[idx];
	int r=val&0xFF, g=val>>8&0xFF, b=val>>16&0xFF;
	dst[idx]=CVT_I2D(r+g+g+b-512);
}
__kernel void tensor2fp32(__global const DataType *src, __global float *dst)
{
	int idx=get_global_id(0);
	dst[idx]=CVT_D2F(src[idx]);
}
__kernel void cc2d(CONST int *indices, CONST DataType *filt, __global const DataType *src, __global DataType *dst)
{
	int idx=get_global_id(0);
	int iw=indices[0], ih=indices[1];
	int kx=idx%iw, ky=idx/iw;
	DataType sum=0;
	if((unsigned)(ky-1)<(unsigned)ih)
	{
		if((unsigned)(kx-1)<(unsigned)iw)
			sum+=MUL(filt[0], src[idx-iw-1]);
		sum+=MUL(filt[1], src[idx-iw]);
		if((unsigned)(kx+1)<(unsigned)iw)
			sum+=MUL(filt[2], src[idx-iw+1]);
	}
	if((unsigned)(kx-1)<(unsigned)iw)
		sum+=MUL(filt[3], src[idx-1]);
	sum+=MUL(filt[4], src[idx]);
	if((unsigned)(kx+1)<(unsigned)iw)
		sum+=MUL(filt[5], src[idx+1]);
	if((unsigned)(ky+1)<(unsigned)ih)
	{
		if((unsigned)(kx-1)<(unsigned)iw)
			sum+=MUL(filt[6], src[idx+iw-1]);
		sum+=MUL(filt[7], src[idx+iw]);
		if((unsigned)(kx+1)<(unsigned)iw)
			sum+=MUL(filt[8], src[idx+iw+1]);
	}
	dst[idx]=sum;
}