#include"runtime.h"
#include<stdio.h>
#include<stdarg.h>
#include<stdlib.h>
#include<math.h>
#ifdef __linux__
#include<time.h>
#else
#include<Windows.h>
#endif

//debug
void			breakpoint(const char *file, int line)
{
	static int count=0;
	printf("breakpoint %d at\n%s(%d)\n\n", count, file, line);
	++count;
}

//utility
char			g_buf[G_BUF_SIZE];
double			time_sec()
{
#ifdef __linux__
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec+1e-9*ts.tv_nsec;
#else
	LARGE_INTEGER ticks, freq;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&ticks);
	return (double)ticks.QuadPart/freq.QuadPart;
#endif
}

//math
long long		maximum(long long a, long long b){return a>b?a:b;}
long long		minimum(long long a, long long b){return a<b?a:b;}
long long		mod(long long x, long long n)
{
	x%=n;
	x+=n&-(x<0);
	return x;
}
int				first_set_bit(unsigned long long n)//idx of LSB
{
	int sh=((n&((1ULL<<32)-1))==0)<<5,	idx =sh;	n>>=sh;
		sh=((n&((1   <<16)-1))==0)<<4,	idx+=sh;	n>>=sh;
		sh=((n&((1   << 8)-1))==0)<<3,	idx+=sh;	n>>=sh;
		sh=((n&((1   << 4)-1))==0)<<2,	idx+=sh;	n>>=sh;
		sh=((n&((1   << 2)-1))==0)<<1,	idx+=sh;	n>>=sh;
		sh= (n&((1   << 1)-1))==0,		idx+=sh;
	return idx;
}
int				first_set_bit16(unsigned short n)//idx of LSB
{
	int sh=((n&((1<<8)-1))==0)<<3,	idx =sh;	n>>=sh;
		sh=((n&((1<<4)-1))==0)<<2,	idx+=sh;	n>>=sh;
		sh=((n&((1<<2)-1))==0)<<1,	idx+=sh;	n>>=sh;
		sh= (n&((1<<1)-1))==0,		idx+=sh;
	return idx;
}
int				floor_log2(unsigned long long n)//idx of MSB
{
	int sh=(n>=1ULL	<<32)<<5,	logn =sh; n>>=sh;
		sh=(n>=1	<<16)<<4;	logn+=sh, n>>=sh;
		sh=(n>=1	<< 8)<<3;	logn+=sh, n>>=sh;
		sh=(n>=1	<< 4)<<2;	logn+=sh, n>>=sh;
		sh=(n>=1	<< 2)<<1;	logn+=sh, n>>=sh;
		sh= n>=1	<< 1;		logn+=sh;
	return logn;
}
int				ceil_log2(unsigned long long n)
{
	int sh=(n>1ULL<<31)<<5,	logn =sh; n>>=sh;
		sh=(n>1   <<15)<<4;	logn+=sh, n>>=sh;
		sh=(n>1   << 7)<<3;	logn+=sh, n>>=sh;
		sh=(n>1   << 3)<<2;	logn+=sh, n>>=sh;
		sh=(n>1   << 1)<<1;	logn+=sh, n>>=sh;
		sh= n>1   << 0;		logn+=sh;
	return logn;
}
int				floor_log10(double x)
{
	static const double pmask[]=//positive powers
	{
		1, 10,		//10^2^0
		1, 100,		//10^2^1
		1, 1e4,		//10^2^2
		1, 1e8,		//10^2^3
		1, 1e16,	//10^2^4
		1, 1e32,	//10^2^5
		1, 1e64,	//10^2^6
		1, 1e128,	//10^2^7
		1, 1e256	//10^2^8
	};
	static const double nmask[]=//negative powers
	{
		1, 0.1,		//1/10^2^0
		1, 0.01,	//1/10^2^1
		1, 1e-4,	//1/10^2^2
		1, 1e-8,	//1/10^2^3
		1, 1e-16,	//1/10^2^4
		1, 1e-32,	//1/10^2^5
		1, 1e-64,	//1/10^2^6
		1, 1e-128,	//1/10^2^7
		1, 1e-256	//1/10^2^8
	};
	int logn, sh;
	if(x<=0)
		return 0x80000000;
	if(x>=1)
	{
		logn=0;
		sh=(x>=pmask[17])<<8;	logn+=sh, x*=nmask[16+(sh!=0)];//x>=1
		sh=(x>=pmask[15])<<7;	logn+=sh, x*=nmask[14+(sh!=0)];
		sh=(x>=pmask[13])<<6;	logn+=sh, x*=nmask[12+(sh!=0)];
		sh=(x>=pmask[11])<<5;	logn+=sh, x*=nmask[10+(sh!=0)];
		sh=(x>=pmask[9])<<4;	logn+=sh, x*=nmask[8+(sh!=0)];
		sh=(x>=pmask[7])<<3;	logn+=sh, x*=nmask[6+(sh!=0)];
		sh=(x>=pmask[5])<<2;	logn+=sh, x*=nmask[4+(sh!=0)];
		sh=(x>=pmask[3])<<1;	logn+=sh, x*=nmask[2+(sh!=0)];
		sh= x>=pmask[1];		logn+=sh;
		return logn;
	}
	logn=-1;
	sh=(x<nmask[17])<<8;	logn-=sh;	x*=pmask[16+(sh!=0)];//x<1
	sh=(x<nmask[15])<<7;	logn-=sh;	x*=pmask[14+(sh!=0)];
	sh=(x<nmask[13])<<6;	logn-=sh;	x*=pmask[12+(sh!=0)];
	sh=(x<nmask[11])<<5;	logn-=sh;	x*=pmask[10+(sh!=0)];
	sh=(x<nmask[9])<<4;		logn-=sh;	x*=pmask[8+(sh!=0)];
	sh=(x<nmask[7])<<3;		logn-=sh;	x*=pmask[6+(sh!=0)];
	sh=(x<nmask[5])<<2;		logn-=sh;	x*=pmask[4+(sh!=0)];
	sh=(x<nmask[3])<<1;		logn-=sh;	x*=pmask[2+(sh!=0)];
	sh= x<nmask[1];			logn-=sh;
	return logn;
}
double			power(double x, int y)
{
	double mask[]={1, 0}, product=1;
	if(y<0)
		mask[1]=1/x, y=-y;
	else
		mask[1]=x;
	for(;;)
	{
		product*=mask[y&1], y>>=1;	//67.7
		if(!y)
			return product;
		mask[1]*=mask[1];
	}
	return product;
}
double			_10pow(int n)
{
	static double *mask=0;
	int k;
	if(!mask)
	{
		mask=(double*)malloc(616*sizeof(double));
		for(k=-308;k<308;++k)		//23.0
			mask[k+308]=power(10., k);
		//	mask[k+308]=exp(k*_ln10);//inaccurate
	}
	if(n<-308)
		return 0;
	if(n>307)
		return _HUGE;
	return mask[n+308];
}

//control
void			prompt(const char *format, ...)
{
	if(format)
	{
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
	}
	printf("\nEnter 0 to continue... ");
	int k=0;
	scanf_s("%d", &k);
	printf("\n");
}
void			exit_success()
{
	printf("\nDone. Enter 0 to exit.\n");
	int k=0;
	scanf_s("%d", &k);
	exit(0);
}
bool			exit_failure(const char *file, int line, int code, const char *condition, const char *format, ...)
{
	printf("%s(%d):", file, line);
	if(condition)
		printf(" ( %s ) == false\n", condition);
	printf("\n");
	if(format)
	{
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		printf("\n");
	}
	printf("\nCRASH\n");
	int k=0;
	scanf_s("%d", &k);
	exit(code);
	return false;
}

#ifndef __linux__
int				set_console_buffer_size(short w, short h)
{
	COORD coords={w, h};
	HANDLE handle=GetStdHandle(STD_OUTPUT_HANDLE);
	int success=SetConsoleScreenBufferSize(handle, coords);
	if(!success)
		printf("Failed to resize console buffer: %d\n\n", GetLastError());
	return success;
}
#endif
