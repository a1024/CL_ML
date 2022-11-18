#pragma once
#ifndef INC_ACME_STDIO_H
#define INC_ACME_STDIO_H
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include<stdio.h>

#ifndef _MSC_VER
#define			sprintf_s	snprintf
#define			vsprintf_s	vsnprintf
#endif

#endif//INC_ACME_STDIO_H
