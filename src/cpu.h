#pragma once
#ifndef INC_AWM_CPU_H
#define INC_AWM_CPU_H

#ifdef _MSC_VER
# if defined _M_AMD64
#  define CPU_X86	64
# elif defined _M_IX86
#  define CPU_X86	32
# elif defined _M_ARM64
#  define CPU_ARM	64
# else
#  error Unknown processor
# endif
#elif defined __GNUC__
# if defined __x86_64__
#  define CPU_X86	64
# elif defined __i386__
#  define CPU_X86	32
# elif defined __aarch64__
#  define CPU_ARM	64
# else
#  error Unknown processor
#endif
#else
#error Unknown compiler
#endif

#endif//INC_AWM_CPU_H