#pragma once
#ifndef INC_UTIL_H
#define INC_UTIL_H
#include<stddef.h>
#ifdef __cplusplus
extern "C"
{
#endif

#define COUNTOF(ARR)		(sizeof(ARR)/sizeof(*(ARR)))
#define BETWEEN(LO, X, HI)	((unsigned)((X)-LO)<(unsigned)(HI+1-LO))
#ifdef _MSC_VER
#define	ALIGN(X)			__declspec(align(X))
#elif defined __GNUC__
#define	ALIGN(X)			__attribute__((aligned(X)))
#endif

int		minimum(int a, int b);
int		maximum(int a, int b);
float	minimumf(float a, float b);
float	maximumf(float a, float b);
double	minimumd(double a, double b);
double	maximumd(double a, double b);
int		clamp(int lo, int x, int hi);
float	clampf(float lo, float x, float hi);
double	clampd(double lo, double x, double hi);
int		floor_log2(unsigned long long n);
int		ceil_log2(unsigned long long n);
int		floor_log10(double x);
double	power(double x, int y);
double	_10pow(int n);

void	memfill(void *dst, const void *src, size_t dstbytes, size_t srcbytes);
void	memswap_slow(void *p1, void *p2, size_t size);
void	memswap(void *p1, void *p2, size_t size, void *temp);
#if defined CPU_X86
void 	memswap_sse(void *p1, void *p2, size_t size);//untested		X  what if size<16?
#endif
void	memreverse(void *p, size_t count, size_t esize);//calls memswap
void	memrotate(void *p, size_t byteoffset, size_t bytesize, void *temp);//temp buffer is min(byteoffset, bytesize-byteoffset)
int		binary_search(const void *base, size_t count, size_t esize, int (*threeway)(const void*, const void*), const void *val, size_t *idx);//returns true if found, otherwise the idx is where val should be inserted, standard bsearch doesn't do this
void	isort(void *base, size_t count, size_t esize, int (*threeway)(const void*, const void*));//binary insertion sort

typedef enum GetOptRetEnum
{
	OPT_ENDOFARGS=-3,
	OPT_INVALIDARG,
	OPT_NOMATCH,
} GetOptRet;
int		acme_getopt(int argc, char **argv, int *start, const char **keywords, int kw_count);//keywords[i]: shortform char, followed by longform null-terminated string, returns 


int				acme_isdigit(char c, char base);

double			time_ms();

#ifdef __cplusplus
}
#endif
#endif//INC_UTIL_H
