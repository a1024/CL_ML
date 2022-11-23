#include<string.h>//memcpy, memmove
#include<math.h>
#include"util.h"
#include"cpu.h"
#if defined CPU_X86
#include<tmmintrin.h>
#endif
#if defined _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include<Windows.h>//QueryPerformance...
#else
#include<time.h>//clock_gettime
#endif

int		minimum(int a, int b){return a<b?a:b;}
int		maximum(int a, int b){return a>b?a:b;}
float	minimumf(float a, float b){return a<b?a:b;}
float	maximumf(float a, float b){return a>b?a:b;}
double	minimumd(double a, double b){return a<b?a:b;}
double	maximumd(double a, double b){return a>b?a:b;}
int		clamp(int lo, int x, int hi)
{
	if(x<lo)
		x=lo;
	if(x>hi)
		x=hi;
	return x;
}
float	clampf(float lo, float x, float hi)
{
	if(x<lo)
		x=lo;
	if(x>hi)
		x=hi;
	return x;
}
double	clampd(double lo, double x, double hi)
{
	if(x<lo)
		x=lo;
	if(x>hi)
		x=hi;
	return x;
}
int		floor_log2(unsigned long long n)
{
	int logn=0;
	int sh=(n>=1ULL<<32)<<5;logn+=sh, n>>=sh;
		sh=(n>=1<<16)<<4;	logn+=sh, n>>=sh;
		sh=(n>=1<< 8)<<3;	logn+=sh, n>>=sh;
		sh=(n>=1<< 4)<<2;	logn+=sh, n>>=sh;
		sh=(n>=1<< 2)<<1;	logn+=sh, n>>=sh;
		sh= n>=1<< 1;		logn+=sh;
	return logn;
}
int		ceil_log2(unsigned long long n)
{
	int l2=floor_log2(n);
	l2+=(1ULL<<l2)<n;
	return l2;
}
int		floor_log10(double x)
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
		sh=(x>=pmask[17])<<8;	logn+=sh, x*=nmask[16+(sh!=0)];
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
	sh=(x<nmask[17])<<8;	logn-=sh;	x*=pmask[16+(sh!=0)];
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
double	power(double x, int y)
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
static double mask[616]={0};
double	_10pow(int n)
{
	static double *mask=0;
	int k;
//	const double _ln10=log(10.);
	if(!mask[0])
	{
		for(k=-308;k<308;++k)		//23.0
			mask[k+308]=power(10., k);
		//	mask[k+308]=exp(k*_ln10);//inaccurate
	}
	if(n<-308)
		return 0;
	if(n>307)
#ifdef _MSC_VER
		return _HUGE;
#elif defined __linux__
		return HUGE_VAL;
#endif
	return mask[n+308];
}

void			memfill(void *dst, const void *src, size_t dstbytes, size_t srcbytes)
{
	size_t copied;
	char *d=(char*)dst;
	const char *s=(const char*)src;
	if(dstbytes<srcbytes)
	{
		memcpy(dst, src, dstbytes);
		return;
	}
	copied=srcbytes;
	memcpy(d, s, copied);
	while(copied<<1<=dstbytes)
	{
		memcpy(d+copied, d, copied);
		copied<<=1;
	}
	if(copied<dstbytes)
		memcpy(d+copied, d, dstbytes-copied);
}
void			memswap_slow(void *p1, void *p2, size_t size)
{
	unsigned char *s1=(unsigned char*)p1, *s2=(unsigned char*)p2, *end=s1+size;
	for(;s1<end;++s1, ++s2)
	{
		const unsigned char t=*s1;
		*s1=*s2;
		*s2=t;
	}
}
void 			memswap(void *p1, void *p2, size_t size, void *temp)
{
	memcpy(temp, p1, size);
	memcpy(p1, p2, size);
	memcpy(p2, temp, size);
}
#if defined CPU_X86
void 			memswap_sse(void *p1, void *p2, size_t size)//untested		X  what if size<16?
{
	ptrdiff_t k, s2;//must be signed
	char *b1, *b2;
	float *a, *b;

	b1=(char*)p1;
	b2=(char*)p2;
	s2=size-15;
	for(k=0;k<s2;k+=16)
	{
		a=(float*)(b1+k);
		b=(float*)(b2+k);
		__m128 v1=_mm_loadu_ps(a);
		__m128 v2=_mm_loadu_ps(b);
		_mm_storeu_ps(a, v2);
		_mm_storeu_ps(b, v1);
	}
	size+=15;
	for(;k<(ptrdiff_t)size;++k)
	{
		unsigned char temp=b1[k];
		b1[k]=b2[k];
		b2[k]=temp;
	}
}
#endif
void			memreverse(void *p, size_t count, size_t esize)
{
	size_t totalsize=count*esize;
	unsigned char *s1=(unsigned char*)p, *s2=s1+totalsize-esize;
#ifdef CPU_X86
	while(s1<s2)
	{
		memswap_sse(s1, s2, esize);
		s1+=esize, s2-=esize;
	}
#else
	void *temp=malloc(esize);
	while(s1<s2)
	{
		memswap(s1, s2, esize, temp);
		s1+=esize, s2-=esize;
	}
	free(temp);
#endif
}
void 			memrotate(void *p, size_t byteoffset, size_t bytesize, void *temp)
{
	unsigned char *buf=(unsigned char*)p;

	if(byteoffset<bytesize-byteoffset)
	{
		memcpy(temp, buf, byteoffset);
		memmove(buf, buf+byteoffset, bytesize-byteoffset);
		memcpy(buf+bytesize-byteoffset, temp, byteoffset);
	}
	else
	{
		memcpy(temp, buf+byteoffset, bytesize-byteoffset);
		memmove(buf+bytesize-byteoffset, buf, byteoffset);
		memcpy(buf, temp, bytesize-byteoffset);
	}
}
int 			binary_search(const void *base, size_t count, size_t esize, int (*threeway)(const void*, const void*), const void *val, size_t *idx)
{
	const unsigned char *buf=(const unsigned char*)base;
	ptrdiff_t L=0, R=(ptrdiff_t)count-1, mid;
	int ret;

	while(L<=R)
	{
		mid=(L+R)>>1;
		ret=threeway(buf+mid*esize, val);
		if(ret<0)
			L=mid+1;
		else if(ret>0)
			R=mid-1;
		else
		{
			if(idx)
				*idx=mid;
			return 1;
		}
	}
	if(idx)
		*idx=L+(L<(ptrdiff_t)count&&threeway(buf+L*esize, val)<0);
	return 0;
}
void 			isort(void *base, size_t count, size_t esize, int (*threeway)(const void*, const void*))
{
	unsigned char *buf=(unsigned char*)base;
	size_t k;
	void *temp;

	if(count<2)
		return;

	temp=malloc((count>>1)*esize);
	for(k=1;k<count;++k)
	{
		size_t idx=0;
		binary_search(buf, k, esize, threeway, buf+k*esize, &idx);
		if(idx<k)
			memrotate(buf+idx*esize, (k-idx)*esize, (k+1-idx)*esize, temp);
	}
	free(temp);
}
int				acme_getopt(int argc, char **argv, int *start, const char **keywords, int kw_count)
{
	int k;
	size_t len;
	const char *arg, *cand;

	if(*start>=argc)
		return OPT_ENDOFARGS;
	
	arg=argv[*start];
	len=strlen(arg);
	if(len<=0)
		return OPT_INVALIDARG;
	//len>=1
	if(arg[0]!='-')
		return OPT_NOMATCH;
	++arg, --len;
	if(len<=0)
		return OPT_INVALIDARG;
	//len>=1
	if(arg[0]!='-')//short form (single dash followed by one character)
	{
		if(len!=1)
			return OPT_INVALIDARG;
		//len==1
		for(k=0;k<kw_count;++k)
		{
			cand=keywords[k];
			if(cand[0]==arg[0])
				return k;
		}
	}
	else//long form (double dash followed by a word)
	{
		++arg, --len;
		if(len<=0)
			return OPT_INVALIDARG;
		//len>=1
		for(k=0;k<kw_count;++k)
		{
			cand=keywords[k];
			if(!strcmp(arg, cand+1))
				return k;
		}
	}
	return OPT_NOMATCH;
}

int				acme_isdigit(char c, char base)
{
	switch(base)
	{
	case 2:		return BETWEEN('0', c, '1');
	case 8:		return BETWEEN('0', c, '7');
	case 10:	return BETWEEN('0', c, '9');
	case 16:	return BETWEEN('0', c, '9')||BETWEEN('A', c&0xDF, 'F');
	}
	return 0;
}

double			time_ms()
{
#ifdef _MSC_VER
	static double inv_f=0;
	LARGE_INTEGER li;
	//if(!inv_f)
	//{
		QueryPerformanceFrequency(&li);//<Windows.h>
		inv_f=1/(double)li.QuadPart;
	//}
	QueryPerformanceCounter(&li);
	return 1000.*(double)li.QuadPart*inv_f;
#else
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);//<time.h>
	return t.tv_sec*1000+t.tv_nsec*1e-6;
#endif
}