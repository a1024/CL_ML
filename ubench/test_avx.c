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
void get_cpu_info(CPUInfo *info)
{
	//https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=msvc-170
	//https://en.wikipedia.org/wiki/CPUID?useskin=monobook
	ArrayHandle arr;
	int regs[4], *p;
	int f1_ecx, f1_edx;
	int f7_ebx, f7_ecx;
	__cpuid(regs, 0);
	ARRAY_ALLOC(int[4], arr, 0, 0, 0, 0);
	for(int id=0, nids=regs[0];id<nids;++id)
	{
		__cpuidex(regs, id, 0);
		ARRAY_APPEND(arr, regs, 1, 1, 0);
	}

	p=(int*)array_at(&arr, 0);
	memcpy(info->vendor, p+1, 4);
	memcpy(info->vendor+4, p+3, 4);
	memcpy(info->vendor+8, p+2, 4);
	memset(info->vendor+12, 0, 4);

	if(arr->count>1)
	{
		p=(int*)array_at(&arr, 1);
		f1_ecx=p[2], f1_edx=p[3];
	}
	else
		f1_ecx=0, f1_edx=0;

	if(arr->count>7)
	{
		p=(int*)array_at(&arr, 7);
		f7_ebx=p[1], f7_ecx=p[2];
	}
	else
		f7_ebx=0, f7_ecx=0;

	__cpuid(regs, 0x80000000);
	int nexids=regs[0];
	//if(nexids>=0x80000001)
	//{
	//	__cpuidex(regs, 0x80000001, 0);
	//	int f81_ecx=regs[2], f81_edx=regs[3];
	//}
	if(nexids>=0x80000004)
	{
		__cpuidex(regs, 0x80000002, 0);
		memcpy(info->brand, regs, sizeof(regs));
		__cpuidex(regs, 0x80000003, 0);
		memcpy(info->brand+sizeof(regs), regs, sizeof(regs));
		__cpuidex(regs, 0x80000004, 0);
		memcpy(info->brand+sizeof(regs)*2, regs, sizeof(regs));
		memset(info->brand+sizeof(regs)*3, 0, sizeof(regs));//for struct alignment
	}
	else
		memset(info->brand, 0, sizeof(info->brand));
	//for(unsigned id=0x80000000, nids=regs[0];id<nids;++id)
	//{
	//	__cpuidex(regs, id, 0);
	//	ARRAY_APPEND(arr2, regs, 1, 1, 0);
	//}

	info->mmx=f1_edx>>23&1;
	info->sse=f1_edx>>25&1;
	info->sse2=f1_edx>>26&1;
	info->sse3=f1_ecx&1;
	info->ssse3=f1_ecx>>9&1;
	info->sse4_1=f1_ecx>>19&1;
	info->sse4_2=f1_ecx>>20&1;
	info->fma=f1_ecx>>12&1;
	info->aes=f1_ecx>>25&1;
	info->sha=f7_ebx>>29&1;
	info->avx=f1_ecx>>28&1;
	info->avx2=f7_ebx>>5&1;
	info->avx512F=f7_ebx>>16&1;
	info->avx512PF=f7_ebx>>26&1;
	info->avx512ER=f7_ebx>>27&1;
	info->avx512CD=f7_ebx>>28&1;
	info->f16c=f1_ecx>>29&1;
	info->rdrand=f1_ecx>>30&1;
	info->rdseed=f7_ebx>>18&1;

	array_free(&arr);
}
//int avx_supported()
//{
//	int regs[4];
//	__cpuid(regs, 1);
//	return regs[2]>>28&1;
//}
static const float filt[]=
{
	1.f/16, 1.f/16, 1.f/16,
	1.f/16, 8.f/16, 1.f/16,
	1.f/16, 1.f/16, 1.f/16,
};
int test_avx_fp32(unsigned char *image, int iw, int ih)
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
	__m256 f0=_mm256_set1_ps(filt[0]);
	__m256 f1=_mm256_set1_ps(filt[1]);
	__m256 f2=_mm256_set1_ps(filt[2]);
	__m256 f3=_mm256_set1_ps(filt[3]);
	__m256 f4=_mm256_set1_ps(filt[4]);
	__m256 f5=_mm256_set1_ps(filt[5]);
	__m256 f6=_mm256_set1_ps(filt[6]);
	__m256 f7=_mm256_set1_ps(filt[7]);
	__m256 f8=_mm256_set1_ps(filt[8]);
	for(int it=0;it<100;++it)
	{
		for(int ky=0;ky<ih;++ky)//3x3 cross-correlation		PAD bug
		{
			for(int kx=0;kx<iw-7;kx+=8)
			{
				int idx=iw*ky+kx;
				__m256 val=_mm256_setzero_ps();
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-7))
						val=_mm256_add_ps(val, _mm256_mul_ps(f0, _mm256_loadu_ps(src+idx-iw-1)));
					val=_mm256_add_ps(val, _mm256_mul_ps(f1, _mm256_loadu_ps(src+idx-iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-7))
						val=_mm256_add_ps(val, _mm256_mul_ps(f2, _mm256_loadu_ps(src+idx-iw+1)));
				}
				if((unsigned)(kx-1)<(unsigned)(iw-7))
					val=_mm256_add_ps(val, _mm256_mul_ps(f3, _mm256_loadu_ps(src+idx-1)));
				val=_mm256_add_ps(val, _mm256_mul_ps(f4, _mm256_loadu_ps(src+idx)));
				if((unsigned)(kx+1)<(unsigned)(iw-7))
					val=_mm256_add_ps(val, _mm256_mul_ps(f5, _mm256_loadu_ps(src+idx+1)));
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-7))
						val=_mm256_add_ps(val, _mm256_mul_ps(f6, _mm256_loadu_ps(src+idx+iw-1)));
					val=_mm256_add_ps(val, _mm256_mul_ps(f7, _mm256_loadu_ps(src+idx+iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-7))
						val=_mm256_add_ps(val, _mm256_mul_ps(f8, _mm256_loadu_ps(src+idx+iw+1)));
				}
				_mm256_storeu_ps(dst+idx, val);
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

	printf("fp32\tAVX\t%12.6lfms  %12lldcycles  RMSE %14lf\n", elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}
int test_avx_fp64(unsigned char *image, int iw, int ih)
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
	__m256d f0=_mm256_set1_pd(filt[0]);
	__m256d f1=_mm256_set1_pd(filt[1]);
	__m256d f2=_mm256_set1_pd(filt[2]);
	__m256d f3=_mm256_set1_pd(filt[3]);
	__m256d f4=_mm256_set1_pd(filt[4]);
	__m256d f5=_mm256_set1_pd(filt[5]);
	__m256d f6=_mm256_set1_pd(filt[6]);
	__m256d f7=_mm256_set1_pd(filt[7]);
	__m256d f8=_mm256_set1_pd(filt[8]);
	for(int it=0;it<100;++it)
	{
		for(int ky=0;ky<ih;++ky)//3x3 cross-correlation		PAD bug
		{
			for(int kx=0;kx<iw-3;kx+=4)
			{
				int idx=iw*ky+kx;
				__m256d val=_mm256_setzero_pd();
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-3))
						val=_mm256_add_pd(val, _mm256_mul_pd(f0, _mm256_loadu_pd(src+idx-iw-1)));
					val=_mm256_add_pd(val, _mm256_mul_pd(f1, _mm256_loadu_pd(src+idx-iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-3))
						val=_mm256_add_pd(val, _mm256_mul_pd(f2, _mm256_loadu_pd(src+idx-iw+1)));
				}
				if((unsigned)(kx-1)<(unsigned)(iw-3))
					val=_mm256_add_pd(val, _mm256_mul_pd(f3, _mm256_loadu_pd(src+idx-1)));
				val=_mm256_add_pd(val, _mm256_mul_pd(f4, _mm256_loadu_pd(src+idx)));
				if((unsigned)(kx+1)<(unsigned)(iw-3))
					val=_mm256_add_pd(val, _mm256_mul_pd(f5, _mm256_loadu_pd(src+idx+1)));
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)(iw-3))
						val=_mm256_add_pd(val, _mm256_mul_pd(f6, _mm256_loadu_pd(src+idx+iw-1)));
					val=_mm256_add_pd(val, _mm256_mul_pd(f7, _mm256_loadu_pd(src+idx+iw)));
					if((unsigned)(kx+1)<(unsigned)(iw-3))
						val=_mm256_add_pd(val, _mm256_mul_pd(f8, _mm256_loadu_pd(src+idx+iw+1)));
				}
				_mm256_storeu_pd(dst+idx, val);
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

	printf("fp64\tAVX\t%12.6lfms  %12lldcycles  RMSE %14lf\n", elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}