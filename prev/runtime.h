#pragma once
#ifndef RUNTIME_H
#define RUNTIME_H



//debug
void			breakpoint(const char *file, int line);
#define			BP		breakpoint(file, __LINE__)

//utility
#ifdef __linux__
#define sprintf_s snprintf
#define scanf_s scanf
#define _HUGE HUGE_VAL
typedef unsigned char byte;
#endif
#define			SIZEOF(STATIC_ARRAY)	(sizeof(STATIC_ARRAY)/sizeof(*(STATIC_ARRAY)))
#define			G_BUF_SIZE	1024
extern char		g_buf[G_BUF_SIZE];
double			time_sec();

//math
long long		maximum(long long a, long long b);
long long		minimum(long long a, long long b);
long long		mod(long long x, long long n);
int				first_set_bit(unsigned long long n);
int				first_set_bit16(unsigned short n);//idx of LSB
int				floor_log2(unsigned long long n);//idx of MSB
int				ceil_log2(unsigned long long n);
int				floor_log10(double x);
double			power(double x, int y);
double			_10pow(int n);

//control
void			prompt(const char *format, ...);
void			exit_success();
bool			exit_failure(const char *file, int line, int code, const char *condition, const char *format, ...);
#define			MY_ASSERT(SUCCESS, MESSAGE, ...)		((SUCCESS)!=0||exit_failure(file, __LINE__, 1, #SUCCESS, MESSAGE,##__VA_ARGS__))
#define			CRASH(MESSAGE, ...)						exit_failure(file, __LINE__, 1, nullptr, MESSAGE,##__VA_ARGS__)
#ifdef __linux__
#define			set_console_buffer_size(...)
#else
int				set_console_buffer_size(short w, short h);
#endif

#ifdef __cplusplus
#include<string>
bool			open_text(const char *filename, std::string &data);
#endif

#endif
