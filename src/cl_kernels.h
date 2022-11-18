#ifndef __OPEN_CL__
#include<math.h>
#include<stdio.h>
#define __kernel
#define __global
#define __constant
#define get_global_id(...)	(__VA_ARGS__)
#define max(...)			(__VA_ARGS__)
#endif