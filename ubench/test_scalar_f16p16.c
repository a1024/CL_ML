#include"util.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#ifdef _MSC_VER
#include<intrin.h>
#elif defined __GNUC__
#include<x86intrin.h>
#endif
#define PREC 8
#define MUL(A, B) ((A)*(B)>>PREC)
static const int filt[]=
{
	0x10, 0x10, 0x10,
	0x10, 0x80, 0x10,
	0x10, 0x10, 0x10,
};
int test_scalar_f16p16(unsigned char *image, int iw, int ih)
{
	int res=iw*ih;
	int *buf1=(int*)malloc(res*sizeof(*buf1));
	int *buf2=(int*)malloc(res*sizeof(*buf2));
	if(!buf1||!buf2)
	{
		printf("Allocation error\n");
		return 1;
	}
	for(int k=0;k<res;++k)//load image
	{
		char r=image[k<<2|0], g=image[k<<2|1], b=image[k<<2|2];
		buf1[k]=r+g+g+b-512;
	}

	int *src=buf1, *dst=buf2, *temp;
	double elapsed=time_ms();
	long long cycles=__rdtsc();
	for(int it=0;it<100;++it)
	{
		for(int ky=0, idx=0;ky<ih;++ky)//3x3 cross-correlation
		{
			for(int kx=0;kx<iw;++kx, ++idx)
			{
				int val=0;
				if((unsigned)(ky-1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)iw)
						val+=MUL(filt[0], src[idx-iw-1]);
					val+=MUL(filt[1], src[idx-iw]);
					if((unsigned)(kx+1)<(unsigned)iw)
						val+=MUL(filt[2], src[idx-iw+1]);
				}
				if((unsigned)(kx-1)<(unsigned)iw)
					val+=MUL(filt[3], src[idx-1]);
				val+=MUL(filt[4], src[idx]);
				if((unsigned)(kx+1)<(unsigned)iw)
					val+=MUL(filt[5], src[idx+1]);
				if((unsigned)(ky+1)<(unsigned)ih)
				{
					if((unsigned)(kx-1)<(unsigned)iw)
						val+=MUL(filt[6], src[idx+iw-1]);
					val+=MUL(filt[7], src[idx+iw]);
					if((unsigned)(kx+1)<(unsigned)iw)
						val+=MUL(filt[8], src[idx+iw+1]);
				}
				dst[idx]=val;
			}
		}
		SWAPVAR(src, dst, temp);
	}
	cycles=__rdtsc()-cycles;
	elapsed=time_ms()-elapsed;

	double sum=0;
	for(int k=0;k<res;++k)
	{
		double val=(double)src[k]/512;
		sum+=val*val;
	}
	sum=sqrt(sum);

	printf("f%dp%d\tScalar\t%12.6lfms  %12lldcycles  RMSE %14lf\n", 32-PREC, PREC, elapsed, cycles, sum);

	free(buf1);
	free(buf2);
	return 0;
}