#include"ubench.h"
#include<stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"
int main(int argc, char **argv)
{
	printf("ubench\n");
#ifdef _DEBUG
	const char *fn="C:/Projects/datasets/dataset-kodak/kodim13.png";

	//const char *fn="D:/ML/dataset-kodak-small/13.PNG";
	//const char *fn="D:/ML/dataset-kodak/kodim16.png";
#else
	if(argc!=2)
	{
		printf("Usage:  %s  image\n", argv[0]);
		return 1;
	}
	const char *fn=argv[1];
#endif
	printf("File:  %s\n", fn);
	int iw=0, ih=0;
	unsigned char *image=stbi_load(fn, &iw, &ih, 0, 4);
	if(!image)
	{
		printf("Cannot open %s\n", fn);
		return 1;
	}

	CPUInfo cpuinfo;
	get_cpu_info(&cpuinfo);
	printf("CPU: %s\n", cpuinfo.brand);
	if(cpuinfo.mmx)printf(" MMX");
	if(cpuinfo.sse)printf(" SSE");
	if(cpuinfo.sse2)printf(" SSE2");
	if(cpuinfo.sse3)printf(" SSE3");
	if(cpuinfo.ssse3)printf(" SSSE3");
	if(cpuinfo.sse4_1)printf(" SSSE4.1");
	if(cpuinfo.sse4_2)printf(" SSSE4.2");
	if(cpuinfo.fma)printf(" FMA");
	printf(" ");
	if(cpuinfo.aes)printf(" AES");
	if(cpuinfo.sha)printf(" SHA");
	printf(" ");
	if(cpuinfo.avx)printf(" AVX");
	if(cpuinfo.avx2)printf(" AVX2");
	printf(" ");
	if(cpuinfo.avx512F)printf(" AVX512F");
	if(cpuinfo.avx512PF)printf(" AVX512PF");
	if(cpuinfo.avx512ER)printf(" AVX512ER");
	if(cpuinfo.avx512CD)printf(" AVX512CD");
	printf(" ");
	if(cpuinfo.f16c)printf(" F16C");
	if(cpuinfo.rdrand)printf(" RDRAND");
	if(cpuinfo.rdseed)printf(" RDSEED");
	printf("\n\n");


	test_scalar_f16p16(image, iw, ih);
	printf("\n");

	test_scalar_fp32(image, iw, ih);
	test_sse_fp32(image, iw, ih);
	if(cpuinfo.avx)
		test_avx_fp32(image, iw, ih);
	if(cpuinfo.avx512F)
		test_avx512_fp32(image, iw, ih);
	printf("\n");

	test_scalar_fp64(image, iw, ih);
	test_sse2_fp64(image, iw, ih);
	if(cpuinfo.avx)
		test_avx_fp64(image, iw, ih);
	if(cpuinfo.avx512F)
		test_avx512_fp64(image, iw, ih);
	printf("\n");
	
#ifdef ENABLE_OPENCL
	test_cl(argv[0], image, iw, ih);
	printf("\n");

	print_clinfo();
#endif

	free(image);
	printf("Done.\n");
	pause();
	return 0;
}