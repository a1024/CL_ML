#include"ubench.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<immintrin.h>
#ifdef _MSC_VER
#include<intrin.h>
#elif defined __GNUC__
#include<x86intrin.h>
#endif
static const float filt[]=
{
	1.f/16, 1.f/16, 1.f/16,
	1.f/16, 8.f/16, 1.f/16,
	1.f/16, 1.f/16, 1.f/16,
};
int test_avx512_fp32(unsigned char *image, int iw, int ih)
{
	int res=iw*ih;
	float *buf1=(float*)malloc(res*sizeof(*buf1));
	float *buf2=(float*)malloc(res*sizeof(*buf2));
	if(!buf1||!buf2)
	{
		printf("Allocation error\n");
		return 1;
	}
	float gain=1.f/512;
	for(int k=0;k<res;++k)//load image
	{
		char r=image[k<<2|0], g=image[k<<2|1], b=image[k<<2|2];
		buf1[k]=(r+g+g+b-512)*gain;
	}
	memset(buf2, 0, res*sizeof(float));

	float *src=buf1, *dst=buf2, *temp;
	double elapsed=time_ms();
	long long cycles=__rdtsc();
	__m512 f0=_mm512_set1_ps(filt[0]);
	__m512 f1=_mm512_set1_ps(filt[1]);
	__m512 f2=_mm512_set1_ps(filt[2]);
	__m512 f3=_mm512_set1_ps(filt[3]);
	__m512 f4=_mm512_set1_ps(filt[4]);
	__m512 f5=_mm512_set1_ps(filt[5]);
	__m512 f6=_mm512_set1_ps(filt[6]);
	__m512 f7=_mm512_set1_ps(filt[7]);
	__m512 f8=_mm512_set1_ps(filt[8]);
	for(int it=0;it<100;++it)
	{
		for(int ky=0;ky<ih;++ky)//3x3 cross-correlation		PAD bug
		{
			for(int kx=0;kx<iw-15;kx+=16)
			{
				int idx=iw*ky+kx;
				__m512 val=_mm512_setzero_ps();
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-15))
						val=_mm512_add_ps(val, _mm512_mul_ps(f0, _mm512_loadu_ps(src+idx-iw-1)));
					val=_mm512_add_ps(val, _mm512_mul_ps(f1, _mm512_loadu_ps(src+idx-iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-15))
						val=_mm512_add_ps(val, _mm512_mul_ps(f2, _mm512_loadu_ps(src+idx-iw+1)));
				}
				if((unsigned)(kx-1)<(unsigned)(iw-15))
					val=_mm512_add_ps(val, _mm512_mul_ps(f3, _mm512_loadu_ps(src+idx-1)));
				val=_mm512_add_ps(val, _mm512_mul_ps(f4, _mm512_loadu_ps(src+idx)));
				if((unsigned)(kx+1)<(unsigned)(iw-15))
					val=_mm512_add_ps(val, _mm512_mul_ps(f5, _mm512_loadu_ps(src+idx+1)));
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-15))
						val=_mm512_add_ps(val, _mm512_mul_ps(f6, _mm512_loadu_ps(src+idx+iw-1)));
					val=_mm512_add_ps(val, _mm512_mul_ps(f7, _mm512_loadu_ps(src+idx+iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-15))
						val=_mm512_add_ps(val, _mm512_mul_ps(f8, _mm512_loadu_ps(src+idx+iw+1)));
				}
				_mm512_storeu_ps(dst+idx, val);
			}
		}
		SWAPVAR(src, dst, temp);
	}
	cycles=__rdtsc()-cycles;
	elapsed=time_ms()-elapsed;

	double sum=0;
	for(int k=0;k<res;++k)
	{
		double val=src[k];
		sum+=val*val;
	}
	sum=sqrt(sum);

	printf("fp32\tAVX-512\t%12.6lfms  %12lldcycles  RMSE %14lf\n", elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}
int test_avx512_fp64(unsigned char *image, int iw, int ih)
{
	int res=iw*ih;
	double *buf1=(double*)malloc(res*sizeof(*buf1));
	double *buf2=(double*)malloc(res*sizeof(*buf2));
	if(!buf1||!buf2)
	{
		printf("Allocation error\n");
		return 1;
	}
	float gain=1./512;
	for(int k=0;k<res;++k)//load image
	{
		char r=image[k<<2|0], g=image[k<<2|1], b=image[k<<2|2];
		buf1[k]=(r+g+g+b-512)*gain;
	}
	memset(buf2, 0, res*sizeof(double));

	double *src=buf1, *dst=buf2, *temp;
	double elapsed=time_ms();
	long long cycles=__rdtsc();
	__m512d f0=_mm512_set1_pd(filt[0]);
	__m512d f1=_mm512_set1_pd(filt[1]);
	__m512d f2=_mm512_set1_pd(filt[2]);
	__m512d f3=_mm512_set1_pd(filt[3]);
	__m512d f4=_mm512_set1_pd(filt[4]);
	__m512d f5=_mm512_set1_pd(filt[5]);
	__m512d f6=_mm512_set1_pd(filt[6]);
	__m512d f7=_mm512_set1_pd(filt[7]);
	__m512d f8=_mm512_set1_pd(filt[8]);
	for(int it=0;it<100;++it)
	{
		for(int ky=0;ky<ih;++ky)//3x3 cross-correlation		PAD bug
		{
			for(int kx=0;kx<iw-7;kx+=8)
			{
				int idx=iw*ky+kx;
				__m512d val=_mm512_setzero_pd();
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-7))
						val=_mm512_add_pd(val, _mm512_mul_pd(f0, _mm512_loadu_pd(src+idx-iw-1)));
					val=_mm512_add_pd(val, _mm512_mul_pd(f1, _mm512_loadu_pd(src+idx-iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-7))
						val=_mm512_add_pd(val, _mm512_mul_pd(f2, _mm512_loadu_pd(src+idx-iw+1)));
				}
				if((unsigned)(kx-1)<(unsigned)(iw-7))
					val=_mm512_add_pd(val, _mm512_mul_pd(f3, _mm512_loadu_pd(src+idx-1)));
				val=_mm512_add_pd(val, _mm512_mul_pd(f4, _mm512_loadu_pd(src+idx)));
				if((unsigned)(kx+1)<(unsigned)(iw-7))
					val=_mm512_add_pd(val, _mm512_mul_pd(f5, _mm512_loadu_pd(src+idx+1)));
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-7))
						val=_mm512_add_pd(val, _mm512_mul_pd(f6, _mm512_loadu_pd(src+idx+iw-1)));
					val=_mm512_add_pd(val, _mm512_mul_pd(f7, _mm512_loadu_pd(src+idx+iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-7))
						val=_mm512_add_pd(val, _mm512_mul_pd(f8, _mm512_loadu_pd(src+idx+iw+1)));
				}
				_mm512_storeu_pd(dst+idx, val);
			}
		}
		SWAPVAR(src, dst, temp);
	}
	cycles=__rdtsc()-cycles;
	elapsed=time_ms()-elapsed;

	double sum=0;
	for(int k=0;k<res;++k)
	{
		double val=src[k];
		sum+=val*val;
	}
	sum=sqrt(sum);

	printf("fp64\tAVX-512\t%12.6lfms  %12lldcycles  RMSE %14lf\n", elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}