# CMakeList.txt : CMake project for ubench, include source and define project specific logic here.

include_directories(
	"E:/C/OpenCL-Headers"
	"C:/Projects/a/OpenCL-Headers"
)
link_directories(
	"E:/C/OpenCL-Headers"
	"C:/Projects/a/OpenCL-Headers"
)
add_executable(ubench
	test_avx.c
	test_avx512.c
	test_cl.c
	test_scalar_f16p16.c
	test_scalar_fp32.c
	test_scalar_fp64.c
	test_sse2.c
	ubench.c
	util.c
)

#if (CMAKE_VERSION VERSION_GREATER 3.12)
#  set_property(TARGET ubench PROPERTY CXX_STANDARD 20)
#endif()
