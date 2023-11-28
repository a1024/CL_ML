#include"util.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<tmmintrin.h>
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
int test_sse_fp32(unsigned char *image, int iw, int ih)
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
	__m128 f0=_mm_set1_ps(filt[0]);
	__m128 f1=_mm_set1_ps(filt[1]);
	__m128 f2=_mm_set1_ps(filt[2]);
	__m128 f3=_mm_set1_ps(filt[3]);
	__m128 f4=_mm_set1_ps(filt[4]);
	__m128 f5=_mm_set1_ps(filt[5]);
	__m128 f6=_mm_set1_ps(filt[6]);
	__m128 f7=_mm_set1_ps(filt[7]);
	__m128 f8=_mm_set1_ps(filt[8]);
	for(int it=0;it<100;++it)
	{
		for(int ky=0;ky<ih;++ky)//3x3 cross-correlation		PAD bug
		{
			for(int kx=0;kx<iw-3;kx+=4)
			{
				int idx=iw*ky+kx;
				__m128 val=_mm_setzero_ps();
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-3))
						val=_mm_add_ps(val, _mm_mul_ps(f0, _mm_loadu_ps(src+idx-iw-1)));
					val=_mm_add_ps(val, _mm_mul_ps(f1, _mm_loadu_ps(src+idx-iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-3))
						val=_mm_add_ps(val, _mm_mul_ps(f2, _mm_loadu_ps(src+idx-iw+1)));
				}
				if((unsigned)(kx-1)<(unsigned)(iw-3))
					val=_mm_add_ps(val, _mm_mul_ps(f3, _mm_loadu_ps(src+idx-1)));
				val=_mm_add_ps(val, _mm_mul_ps(f4, _mm_loadu_ps(src+idx)));
				if((unsigned)(kx+1)<(unsigned)(iw-3))
					val=_mm_add_ps(val, _mm_mul_ps(f5, _mm_loadu_ps(src+idx+1)));
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-3))
						val=_mm_add_ps(val, _mm_mul_ps(f6, _mm_loadu_ps(src+idx+iw-1)));
					val=_mm_add_ps(val, _mm_mul_ps(f7, _mm_loadu_ps(src+idx+iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-3))
						val=_mm_add_ps(val, _mm_mul_ps(f8, _mm_loadu_ps(src+idx+iw+1)));
				}
				//for(int k=0;k<4;++k)//
				//{
				//	if(fabsf(val.m128_f32[k])>1e6f)
				//		printf("");
				//}
				_mm_storeu_ps(dst+idx, val);
			}
		}
		SWAPVAR(src, dst, temp);
	}
	cycles=__rdtsc()-cycles;
	elapsed=time_ms()-elapsed;
	
	//printf("0x%08X\n", *(int*)src);//

	double sum=0;
	for(int k=0;k<res;++k)
	{
		double val=src[k];
		if(fabs(val)>1e6)
			printf("XY %5d %5d  0x%08X %f\n", k%iw, k/iw, *(int*)(src+k), src[k]);
		//if(*(int*)(src+k)==0xCDCDCDCD)
		//	printf("Error");
		sum+=val*val;
	}
	sum=sqrt(sum);

	printf("fp32\tSSE\t%12.6lfms  %12lldcycles  RMSE %14lf\n", elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}
int test_sse2_fp64(unsigned char *image, int iw, int ih)
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
	memset(buf2, 0, res*sizeof(*buf2));

	double *src=buf1, *dst=buf2, *temp;
	double elapsed=time_ms();
	long long cycles=__rdtsc();
	__m128d f0=_mm_set1_pd(filt[0]);
	__m128d f1=_mm_set1_pd(filt[1]);
	__m128d f2=_mm_set1_pd(filt[2]);
	__m128d f3=_mm_set1_pd(filt[3]);
	__m128d f4=_mm_set1_pd(filt[4]);
	__m128d f5=_mm_set1_pd(filt[5]);
	__m128d f6=_mm_set1_pd(filt[6]);
	__m128d f7=_mm_set1_pd(filt[7]);
	__m128d f8=_mm_set1_pd(filt[8]);
	for(int it=0;it<100;++it)
	{
		for(int ky=0;ky<ih;++ky)//3x3 cross-correlation		PAD bug
		{
			for(int kx=0;kx<iw-1;kx+=2)
			{
				int idx=iw*ky+kx;
				__m128d val=_mm_setzero_pd();
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-1))
						val=_mm_add_pd(val, _mm_mul_pd(f0, _mm_loadu_pd(src+idx-iw-1)));
					val=_mm_add_pd(val, _mm_mul_pd(f1, _mm_loadu_pd(src+idx-iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-1))
						val=_mm_add_pd(val, _mm_mul_pd(f2, _mm_loadu_pd(src+idx-iw+1)));
				}
				if((unsigned)(kx-1)<(unsigned)(iw-1))
					val=_mm_add_pd(val, _mm_mul_pd(f3, _mm_loadu_pd(src+idx-1)));
				val=_mm_add_pd(val, _mm_mul_pd(f4, _mm_loadu_pd(src+idx)));
				if((unsigned)(kx+1)<(unsigned)(iw-1))
					val=_mm_add_pd(val, _mm_mul_pd(f5, _mm_loadu_pd(src+idx+1)));
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-1))
						val=_mm_add_pd(val, _mm_mul_pd(f6, _mm_loadu_pd(src+idx+iw-1)));
					val=_mm_add_pd(val, _mm_mul_pd(f7, _mm_loadu_pd(src+idx+iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-1))
						val=_mm_add_pd(val, _mm_mul_pd(f8, _mm_loadu_pd(src+idx+iw+1)));
				}
				_mm_storeu_pd(dst+idx, val);
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

	printf("fp64\tSSE2\t%12.6lfms  %12lldcycles  RMSE %14lf\n", elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}