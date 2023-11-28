#include"ubench.h"
#ifdef ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 300
#include<CL/cl.h>
#include<CL/cl_ext.h>
#endif
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<ctype.h>
#include<math.h>
#if defined ENABLE_OPENCL
#ifdef _MSC_VER
#include<Windows.h>
//#pragma comment(lib, "OpenCL.lib")
#elif defined __linux__
#include<dlfcn.h>
#endif
#endif
#ifdef _MSC_VER
#include<intrin.h>
#elif defined __GNUC__
#include<x86intrin.h>
#endif
static const char file[]=__FILE__;

#ifdef ENABLE_OPENCL
const char *clerr2str(int error)
{
	const char *a=0;
#define 		EC(x)		case x:a=(const char*)#x;break;
#define 		EC2(n, x)	case n:a=(const char*)#x;break;
	switch(error)
	{
	EC(CL_SUCCESS)
	EC(CL_DEVICE_NOT_FOUND)
	EC(CL_DEVICE_NOT_AVAILABLE)
	EC(CL_COMPILER_NOT_AVAILABLE)
	EC(CL_MEM_OBJECT_ALLOCATION_FAILURE)
	EC(CL_OUT_OF_RESOURCES)
	EC(CL_OUT_OF_HOST_MEMORY)
	EC(CL_PROFILING_INFO_NOT_AVAILABLE)
	EC(CL_MEM_COPY_OVERLAP)
	EC(CL_IMAGE_FORMAT_MISMATCH)
	EC(CL_IMAGE_FORMAT_NOT_SUPPORTED)
	EC(CL_BUILD_PROGRAM_FAILURE)
	EC(CL_MAP_FAILURE)
//#ifdef CL_VERSION_1_1
	EC(CL_MISALIGNED_SUB_BUFFER_OFFSET)
	EC(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
//#endif
//#ifdef CL_VERSION_1_2
	EC(CL_COMPILE_PROGRAM_FAILURE)
	EC(CL_LINKER_NOT_AVAILABLE)
	EC(CL_LINK_PROGRAM_FAILURE)
	EC(CL_DEVICE_PARTITION_FAILED)
	EC(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
//#endif
	EC(CL_INVALID_VALUE)
	EC(CL_INVALID_DEVICE_TYPE)
	EC(CL_INVALID_PLATFORM)
	EC(CL_INVALID_DEVICE)
	EC(CL_INVALID_CONTEXT)
	EC(CL_INVALID_QUEUE_PROPERTIES)
	EC(CL_INVALID_COMMAND_QUEUE)
	EC(CL_INVALID_HOST_PTR)
	EC(CL_INVALID_MEM_OBJECT)
	EC(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
	EC(CL_INVALID_IMAGE_SIZE)
	EC(CL_INVALID_SAMPLER)
	EC(CL_INVALID_BINARY)
	EC(CL_INVALID_BUILD_OPTIONS)
	EC(CL_INVALID_PROGRAM)
	EC(CL_INVALID_PROGRAM_EXECUTABLE)
	EC(CL_INVALID_KERNEL_NAME)
	EC(CL_INVALID_KERNEL_DEFINITION)
	EC(CL_INVALID_KERNEL)
	EC(CL_INVALID_ARG_INDEX)
	EC(CL_INVALID_ARG_VALUE)
	EC(CL_INVALID_ARG_SIZE)
	EC(CL_INVALID_KERNEL_ARGS)
	EC(CL_INVALID_WORK_DIMENSION)
	EC(CL_INVALID_WORK_GROUP_SIZE)
	EC(CL_INVALID_WORK_ITEM_SIZE)
	EC(CL_INVALID_GLOBAL_OFFSET)
	EC(CL_INVALID_EVENT_WAIT_LIST)
	EC(CL_INVALID_EVENT)
	EC(CL_INVALID_OPERATION)
	EC(CL_INVALID_GL_OBJECT)
	EC(CL_INVALID_BUFFER_SIZE)
	EC(CL_INVALID_MIP_LEVEL)
	EC(CL_INVALID_GLOBAL_WORK_SIZE)
//#ifdef CL_VERSION_1_1
	EC(CL_INVALID_PROPERTY)
//#endif
//#ifdef CL_VERSION_1_2
	EC(CL_INVALID_IMAGE_DESCRIPTOR)
	EC(CL_INVALID_COMPILER_OPTIONS)
	EC(CL_INVALID_LINKER_OPTIONS)
	EC(CL_INVALID_DEVICE_PARTITION_COUNT)
//#endif
//#ifdef CL_VERSION_2_0
	EC2(-69, CL_INVALID_PIPE_SIZE)
	EC2(-70, CL_INVALID_DEVICE_QUEUE)
//#endif
//#ifdef CL_VERSION_2_2
	EC2(-71, CL_INVALID_SPEC_ID)
	EC2(-72, CL_MAX_SIZE_RESTRICTION_EXCEEDED)
//#endif
//	EC(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR)
//	EC(CL_PLATFORM_NOT_FOUND_KHR)
	EC2(-1002, CL_INVALID_D3D10_DEVICE_KHR)
	EC2(-1003, CL_INVALID_D3D10_RESOURCE_KHR)
	EC2(-1004, CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR)
	EC2(-1005, CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR)
	EC2(-1006, CL_INVALID_D3D11_DEVICE_KHR)
	EC2(-1007, CL_INVALID_D3D11_RESOURCE_KHR)
	EC2(-1008, CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR)
	EC2(-1009, CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR)
#ifndef __linux__
	EC2(-1010, CL_INVALID_D3D9_DEVICE_NV_or_CL_INVALID_DX9_DEVICE_INTEL)
	EC2(-1011, CL_INVALID_D3D9_RESOURCE_NV_or_CL_INVALID_DX9_RESOURCE_INTEL)
	EC2(-1012, CL_D3D9_RESOURCE_ALREADY_ACQUIRED_NV_or_CL_DX9_RESOURCE_ALREADY_ACQUIRED_INTEL)
	EC2(-1013, CL_D3D9_RESOURCE_NOT_ACQUIRED_NV_or_CL_DX9_RESOURCE_NOT_ACQUIRED_INTEL)
#endif
	EC2(-1092, CL_EGL_RESOURCE_NOT_ACQUIRED_KHR)
	EC2(-1093, CL_INVALID_EGL_OBJECT_KHR)
	EC2(-1094, CL_INVALID_ACCELERATOR_INTEL)
	EC2(-1095, CL_INVALID_ACCELERATOR_TYPE_INTEL)
	EC2(-1096, CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL)
	EC2(-1097, CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL)
	EC2(-1098, CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL)
	EC2(-1099, CL_INVALID_VA_API_MEDIA_SURFACE_INTEL)
	EC2(-1101, CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL)
	case 1:a="File failure";break;//
	default:
		a="???";
		break;
	}
#undef			EC
#undef			EC2
	return a;
}
#define CHECK(E, MSG, ...) (!(E)||log_error(file, __LINE__, 1, MSG, ##__VA_ARGS__))
#define CHECKCL(E, FUNC) (!(E)||log_error(file, __LINE__, 1, "OpenCL ERROR  %s  %d  %s", #FUNC, E, clerr2str(E)))
#define WARNCL(E, FUNC) (!(E)||log_error(file, __LINE__, 0, "OpenCL ERROR  %s  %d  %s", #FUNC, E, clerr2str(E)))

typedef struct APIOpenCLStruct
{
	cl_int (__stdcall *GetPlatformIDs)(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);//1.0
	cl_int (__stdcall *GetPlatformInfo)(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);//1.0
	cl_int (__stdcall *GetDeviceIDs)(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);//1.0
	cl_int (__stdcall *GetDeviceInfo)(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);//1.0
	cl_context (__stdcall *CreateContext)(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (__stdcall *pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data), void *user_data, cl_int *errcode_ret);//1.0
	cl_int (__stdcall *ReleaseContext)(cl_context context);//1.0

	cl_command_queue (__stdcall *CreateCommandQueue)(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret);//1.2 (deprecated)
	cl_command_queue (__stdcall *CreateCommandQueueWithProperties)(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret);//2.0
	cl_int (__stdcall *ReleaseCommandQueue)(cl_command_queue command_queue);//1.0
	
	cl_program (__stdcall *CreateProgramWithSource)(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);//1.0
	cl_int (__stdcall *BuildProgram)(cl_program program, cl_uint num_devices, const cl_device_id * device_list, const char *options, void (__stdcall *pfn_notify)(cl_program program, void *user_data), void *user_data);//1.0
	cl_int (__stdcall *GetProgramBuildInfo)(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);//1.0
	cl_kernel (__stdcall *CreateKernel)(cl_program program, const char *kernel_name, cl_int *errcode_ret);//1.0
	cl_int (__stdcall *ReleaseProgram)(cl_program program);//1.0
	
	cl_mem (__stdcall *CreateBuffer)(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);//1.0
	cl_int (__stdcall *EnqueueWriteBuffer)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);//1.0
	cl_int (__stdcall *EnqueueReadBuffer)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);//1.0
	cl_int (__stdcall *ReleaseMemObject)(cl_mem memobj);//1.0
	
	cl_int (__stdcall *SetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);//1.0
	cl_int (__stdcall *EnqueueNDRangeKernel)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);//1.0

	cl_int (__stdcall *Flush)(cl_command_queue command_queue);//1.0
	cl_int (__stdcall *Finish)(cl_command_queue command_queue);//1.0
} APIOpenCL;
void *hOpenCL=0;
static int opencl_loaded=0;
int load_opencl_api(APIOpenCL *cl)
{
	if(!opencl_loaded)
	{
#ifdef _WIN32
		hOpenCL=LoadLibraryA("OpenCL.dll");
#define GETFUNC(X) *(void**)&cl->X=GetProcAddress(hOpenCL, "cl" #X)
#elif defined __linux__
		hOpenCL=dlopen("libOpenCL.so", RTLD_LAZY);
#define GETFUNC(X) *(void**)&cl->X=dlsym(hOpenCL, "cl" #X)
#endif
		if(!hOpenCL)
			return 1;
		GETFUNC(GetPlatformIDs);
		GETFUNC(GetPlatformInfo);
		GETFUNC(GetDeviceIDs);
		GETFUNC(GetDeviceInfo);
		GETFUNC(CreateContext);
		GETFUNC(ReleaseContext);

		GETFUNC(CreateCommandQueue);//1.0 deprecated
		GETFUNC(CreateCommandQueueWithProperties);//2.0
		GETFUNC(ReleaseCommandQueue);//1.0

		GETFUNC(CreateProgramWithSource);
		GETFUNC(BuildProgram);
		GETFUNC(GetProgramBuildInfo);
		GETFUNC(CreateKernel);
		GETFUNC(ReleaseProgram);

		GETFUNC(CreateBuffer);
		GETFUNC(EnqueueWriteBuffer);
		GETFUNC(EnqueueReadBuffer);
		GETFUNC(ReleaseMemObject);

		GETFUNC(SetKernelArg);
		GETFUNC(EnqueueNDRangeKernel);

		GETFUNC(Flush);
		GETFUNC(Finish);
#undef  GETFUNC
		opencl_loaded=1;
	}
	return 0;
}
static APIOpenCL cl={0};

typedef enum AttributeTypeEnum
{
	ATTR_DEVICETYPE,
	ATTR_STR,
	ATTR_STR_LIST,
	ATTR_UINT,
	ATTR_ULONG,
	ATTR_SIZET,
	ATTR_SIZET_ARRAY,
} AttrType;
typedef struct AttributeStruct
{
	const char *name;
	cl_platform_info flag;
	AttrType attrtype;
} Attribute;
unsigned maxdims=0;
Attribute platform_attributes[]=
{
	{"Name                    ", CL_PLATFORM_NAME,				ATTR_STR},
	{"Vendor                  ", CL_PLATFORM_VENDOR,			ATTR_STR},
	{"Version                 ", CL_PLATFORM_VERSION,			ATTR_STR},
	{"Profile                 ", CL_PLATFORM_PROFILE,			ATTR_STR},
	{"Platform ext            ", CL_PLATFORM_EXTENSIONS,			ATTR_STR_LIST},
};
Attribute device_attributes[]=
{
	{"Device type             ", CL_DEVICE_TYPE,				ATTR_DEVICETYPE},
	{"Vendor                  ", CL_DEVICE_VENDOR,				ATTR_STR},
	{"Device version          ", CL_DEVICE_VERSION,				ATTR_STR},
	{"Driver version          ", CL_DRIVER_VERSION,				ATTR_STR},
	{"Device ext              ", CL_DEVICE_EXTENSIONS,			ATTR_STR_LIST},
	{"Max ndim                ", CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,	ATTR_UINT},
	{"Max dims                ", CL_DEVICE_MAX_WORK_ITEM_SIZES,		ATTR_SIZET_ARRAY},//depends on previous attribute
	{"Max workgroup size      ", CL_DEVICE_MAX_WORK_GROUP_SIZE,		ATTR_SIZET},
	{"Char   preferred width  ", CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,	ATTR_UINT},
	{"Short  preferred width  ", CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,	ATTR_UINT},
	{"Int    preferred width  ", CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,	ATTR_UINT},
	{"Half   preferred width  ", CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,	ATTR_UINT},
	{"Float  preferred width  ", CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,	ATTR_UINT},
	{"Double preferred width  ", CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,	ATTR_UINT},
	{"Address bits            ", CL_DEVICE_ADDRESS_BITS,			ATTR_UINT},
};
static void query_print_attr(int is_platform, void *handle, Attribute *attr)
{
	int error=0;
	size_t size=0;
	void *data=0;
	switch(attr->attrtype)
	{
	case ATTR_DEVICETYPE:
		size=sizeof(cl_device_type);
		break;
	case ATTR_STR:
	case ATTR_STR_LIST:
		if(is_platform)
		{
			error=cl.GetPlatformInfo((cl_platform_id)handle, attr->flag, 0, 0, &size);
			CHECKCL(error, clGetPlatformInfo);
		}
		else
		{
			error=cl.GetDeviceInfo((cl_device_id)handle, attr->flag, 0, 0, &size);
			CHECKCL(error, clGetDeviceInfo);
		}
		break;
	case ATTR_UINT:
		size=sizeof(unsigned);
		break;
	case ATTR_ULONG:
		size=sizeof(unsigned long);
		break;
	case ATTR_SIZET:
		size=sizeof(size_t);
		break;
	case ATTR_SIZET_ARRAY:
		if(is_platform)
		{
			error=cl.GetPlatformInfo((cl_platform_id)handle, attr->flag, 0, 0, &size);
			CHECKCL(error, clGetPlatformInfo);
		}
		else
		{
			if(attr->flag==CL_DEVICE_MAX_WORK_ITEM_SIZES)
			{
				error=cl.GetDeviceInfo((cl_device_id)handle, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &size, 0);
				CHECKCL(error, clGetDeviceInfo);
				size*=sizeof(size_t);
			}
			else
			{
				error=cl.GetDeviceInfo((cl_device_id)handle, attr->flag, 0, 0, &size);
				CHECKCL(error, clGetDeviceInfo);
			}
		}
		break;
	default:
		return;
	}
	
	data=malloc(size+16);
	if(!data)
	{
		LOG_ERROR("Allocation error");
		return;
	}

	if(is_platform)
	{
		error=cl.GetPlatformInfo((cl_platform_id)handle, attr->flag, size, data, 0);
		CHECKCL(error, clGetPlatformInfo);
	}
	else
	{
		error=cl.GetDeviceInfo((cl_device_id)handle, attr->flag, size, data, 0);
		CHECKCL(error, clGetDeviceInfo);
	}
	printf("%s", attr->name);
	switch(attr->attrtype)
	{
	case ATTR_DEVICETYPE:
		switch(*(cl_device_type*)data)
		{
		case CL_DEVICE_TYPE_CPU:
			printf("CPU\n");
			break;
		case CL_DEVICE_TYPE_GPU:
			printf("GPU\n");
			break;
		case CL_DEVICE_TYPE_ACCELERATOR:
			printf("Accelerator\n");
			break;
		case CL_DEVICE_TYPE_CUSTOM:
			printf("Custom\n");
			break;
		}
		break;
	case ATTR_STR:
		printf("%s\n", (const char*)data);
		break;
	case ATTR_STR_LIST:
		{
			const char *list=(const char*)data;
			int len=(int)strlen(list);
			printf("\n");
			for(int idx=0;;)
			{
				for(;idx<len&&isspace(list[idx]);++idx);
				int start=idx;
				for(;idx<len&&!isspace(list[idx]);++idx);
				if(idx>=len)
					break;
				printf("\t%.*s\n", idx-start, list+start);
			}
		}
		break;
	case ATTR_UINT:
		printf("%d\n", *(unsigned*)data);
		if(attr->flag==CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
			maxdims=*(unsigned*)data;
		break;
	case ATTR_ULONG:
		printf("%d\n", *(unsigned long*)data);
		break;
	case ATTR_SIZET:
		printf("%lld\n", (long long)*(size_t*)data);
		break;
	case ATTR_SIZET_ARRAY:
		for(int k=0;k<size/sizeof(size_t);++k)
			printf("%lld ", ((size_t*)data)[k]);
		printf("\n");
		break;
	}
	free(data);
}
void print_clinfo()
{
	if(load_opencl_api(&cl))
	{
		printf("OpenCL is NOT SUPPORTED\n");
		return;
	}

	int error=0;
	unsigned nplatforms=0;
	error=cl.GetPlatformIDs(0, 0, &nplatforms);		CHECKCL(error, clGetPlatformIDs);
	if(!nplatforms)
	{
		printf("No OpenCL platforms\n");
		return;
	}
	printf("Found %d platform(s)\n", nplatforms);
	cl_platform_id *platforms=(cl_platform_id*)malloc(nplatforms*sizeof(cl_platform_id*));
	void *temp=0;
	if(!platforms)
	{
		LOG_ERROR("Allocation error");
		return;
	}
	error=cl.GetPlatformIDs(nplatforms, platforms, 0);	CHECKCL(error, clGetPlatformIDs);

	for(unsigned kp=0;kp<nplatforms;++kp)
	{
		cl_platform_id platform=platforms[kp];
		printf("PLATFORM %d:\n", kp);
		for(int ka=0;ka<_countof(platform_attributes);++ka)
			query_print_attr(1, platform, platform_attributes+ka);
		printf("\n");

		unsigned ndevices=0;
		error=cl.GetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, 0, &ndevices);	CHECKCL(error, clGetDeviceIDs);
		if(!ndevices)
		{
			printf("No OpenCL devices on this platform\n");
			continue;
		}
		cl_device_id *devices=(cl_device_id*)malloc(ndevices*sizeof(cl_device_id));
		if(!devices)
		{
			LOG_ERROR("Allocation error");
			return;
		}
		error=cl.GetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ndevices, devices, 0);	CHECKCL(error, clGetDeviceIDs);
		for(int kd=0;kd<(int)ndevices;++kd)
		{
			cl_device_id device=devices[kd];
			printf("DEVICE %d:\n", kd);
			for(int ka=0;ka<_countof(device_attributes);++ka)
				query_print_attr(0, device, device_attributes+ka);
	
			cl_device_fp_config config=0;
			cl.GetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(config), &config, 0);
			if(config)
			{
				printf("cl_khr_fp16 config:\n");
#define CASE(F) if((config&F)==F)printf("\t" #F "\n")
				CASE(CL_FP_DENORM);
				CASE(CL_FP_INF_NAN);
				CASE(CL_FP_ROUND_TO_NEAREST);
				CASE(CL_FP_ROUND_TO_ZERO);
				CASE(CL_FP_ROUND_TO_INF);
#undef  CASE
			}
			else
				printf("cl_khr_fp16 is NOT SUPPORTED\n");
			printf("\n");
		}
		printf("\n");
		free(devices);
	}
	free(platforms);
}


#define KERNELNAMES\
	CLKERNEL(buf2tensor)\
	CLKERNEL(image2tensor)\
	CLKERNEL(tensor2fp32)\
	CLKERNEL(cc2d)
typedef enum CLKernelIdxEnum
{
#define CLKERNEL(LABEL)	OCL_##LABEL,
	KERNELNAMES
#undef  CLKERNEL
	OCL_NKERNELS,
} CLKernelIdx;
const char *kernelnames[]=
{
#define CLKERNEL(LABEL)	#LABEL,
	KERNELNAMES
#undef  CLKERNEL
};
cl_kernel kernels[OCL_NKERNELS]={0};


static const float filt[]=
{
	1.f/16, 1.f/16, 1.f/16,
	1.f/16, 8.f/16, 1.f/16,
	1.f/16, 1.f/16, 1.f/16,
};
int test_cl(const char *programname, unsigned char *image, int iw, int ih)
{
	if(load_opencl_api(&cl))
	{
		printf("OpenCL is NOT SUPPORTED\n");
		return 1;
	}

	int res=iw*ih;

	cl_platform_id platform=0;
	cl_device_id device=0;
	size_t maxlocalsize=0;
	cl_context context=0;
	cl_command_queue commandqueue=0;
	cl_program program=0;
	cl_kernel kernel_cc2d=0;

	int error=0;
	error=cl.GetPlatformIDs(1, &platform, 0);				CHECKCL(error, clGetPlatformIDs);
	error=cl.GetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);	CHECKCL(error, clGetDeviceIDs);
	
	cl_device_fp_config config=0;
	cl.GetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(config), &config, 0);
	int halfprec_supported=config!=0;
	
	//create context & command queue
#ifdef CL_GL_INTEROP
	cl_context_properties properties[8]={};
	if(cl_gl_interop)
	{
		auto gl_context=eglGetCurrentContext();//changes when resuming
		auto egl_display=eglGetCurrentDisplay();
		properties[0]=CL_GL_CONTEXT_KHR,	properties[1]=(cl_context_properties)gl_context;//https://stackoverflow.com/questions/26802905/getting-opengl-buffers-using-opencl
		properties[2]=CL_EGL_DISPLAY_KHR,	properties[3]=(cl_context_properties)egl_display;
		properties[4]=CL_CONTEXT_PLATFORM,	properties[5]=(cl_context_properties)platform;
		properties[6]=0, properties[7]=0;
	}
	else
	{
		properties[0]=CL_CONTEXT_PLATFORM, properties[1]=(cl_context_properties)platform;
		properties[2]=0, properties[3]=0;
	}
	context=cl.CreateContext(properties, 1, &devices[0], 0, 0, &error);	CHECKCL(error, clCreateContext);
#else
	context=cl.CreateContext(0, 1, &device, 0, 0, &error);			CHECKCL(error, clCreateContext);
#endif
	
	if(cl.CreateCommandQueueWithProperties)
	{
		commandqueue=cl.CreateCommandQueueWithProperties(context, device, 0, &error);
		CHECKCL(error, clCreateCommandQueueWithProperties);
	}
	else
	{
		commandqueue=cl.CreateCommandQueue(context, device, 0, &error);
		CHECKCL(error, clCreateCommandQueue);
	}
//#if CL_TARGET_OPENCL_VERSION>=200
//	commandqueue=cl.CreateCommandQueueWithProperties(context, device, 0, &error);	CHECKCL(error, clCreateCommandQueueWithProperties);
//#else
//	commandqueue=cl.CreateCommandQueue(context, device, 0, &error);	CHECKCL(error, clCreateCommandQueue);
//#endif

	float *buf=(float*)malloc(res*sizeof(float));
	if(!buf)
	{
		printf("Allocation error");
		return 1;
	}

	ArrayHandle cwd;
	{//first look in program folder
		STR_COPY(cwd, programname, strlen(programname));
		int k;
		for(k=(int)cwd->count-1;k>=0&&cwd->data[k]!='/'&&cwd->data[k]!='\\';--k);
		memset(cwd->data+k+1, 0, cwd->count-(k+1));
		cwd->count=k+1;
		STR_APPEND(cwd, "kernels.h", 9, 1);
	}
	ArrayHandle srctext=load_file(cwd->data, 0, 0, 0);
	array_free(&cwd);
	if(!srctext)
		srctext=load_file("E:/C/ubench/ubench/kernels.h", 0, 0, 0);
	if(!srctext)
		srctext=load_file("C:/Projects/ubench/ubench/kernels.h", 0, 0, 0);
	if(!srctext)
	{
		printf("Cannot open \'kernels.h\'\n");
		return 1;
	}
	const char *k_src=(const char*)srctext->data;
	size_t k_len=srctext->count;
	for(int kprec=0;kprec<2+halfprec_supported;++kprec)
	{
		const char *testname=0;
		int datatypesize=0;
		switch(kprec)
		{
		case 0://float
			testname="fp32";
			datatypesize=sizeof(float);
			snprintf(g_buf, G_BUF_SIZE, "-D__OPEN_CL__");
			break;
		case 1://int
			testname="f16p16";
			datatypesize=sizeof(int);
			snprintf(g_buf, G_BUF_SIZE, "-D__OPEN_CL__ -DPREC_FIXED=%d", 16);
			break;
		case 2://half
			testname="fp16";
			datatypesize=sizeof(short);
			snprintf(g_buf, G_BUF_SIZE, "-D__OPEN_CL__ -DPREC_HALF");
			break;
		}
		//printf("Test: %s\n", testname);
		
		program=cl.CreateProgramWithSource(context, 1, (const char**)&k_src, &k_len, &error);	WARNCL(error, clCreateProgramWithSource);
		error=cl.BuildProgram(program, 1, &device, g_buf, 0, 0);
		if(error)
		{
			size_t retlen=0;
			error=cl.GetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, G_BUF_SIZE, g_buf, &retlen);
			if(retlen>G_BUF_SIZE)
			{
				char *buf=(char*)malloc(retlen+10);
				error=cl.GetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, retlen+10, buf, &retlen);	WARNCL(error, clGetProgramBuildInfo);
				printf("\nOpenCL Compilation failed:\n%s\n", buf);
				free(buf);
				LOG_ERROR("Aborting");
			}
			else
				LOG_ERROR("\nOpenCL Compilation failed:\n%s\n", g_buf);
		}
		for(int k=0;k<OCL_NKERNELS;++k)
		{
			kernels[k]=cl.CreateKernel(program, kernelnames[k], &error);
			CHECK(error, "Couldn't find kernel %s", kernelnames[k]);
		}
		

		int indices[]={iw, ih};
		cl_mem gpu_srcfilt	=cl.CreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float[9]), 0, &error);	CHECKCL(error, clCreateBuffer);
		cl_mem gpu_image	=cl.CreateBuffer(context, CL_MEM_READ_WRITE, res*sizeof(int), 0, &error);	CHECKCL(error, clCreateBuffer);

		cl_mem gpu_indices	=cl.CreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int[2]), 0, &error);		CHECKCL(error, clCreateBuffer);
		cl_mem gpu_filt		=cl.CreateBuffer(context, CL_MEM_READ_WRITE, 9*datatypesize, 0, &error);		CHECKCL(error, clCreateBuffer);
		cl_mem gpu_src		=cl.CreateBuffer(context, CL_MEM_READ_WRITE, res*datatypesize, 0, &error);	CHECKCL(error, clCreateBuffer);
		cl_mem gpu_dst		=cl.CreateBuffer(context, CL_MEM_READ_WRITE, res*datatypesize, 0, &error);	CHECKCL(error, clCreateBuffer);

		error=cl.EnqueueWriteBuffer(commandqueue, gpu_indices, CL_FALSE, 0, sizeof(indices), indices, 0, 0, 0);	CHECKCL(error, clEnqueueWriteBuffer);
		error=cl.EnqueueWriteBuffer(commandqueue, gpu_srcfilt, CL_FALSE, 0, sizeof(filt), filt, 0, 0, 0);	CHECKCL(error, clEnqueueWriteBuffer);
		error=cl.EnqueueWriteBuffer(commandqueue, gpu_image, CL_FALSE, 0, res*sizeof(int), image, 0, 0, 0);	CHECKCL(error, clEnqueueWriteBuffer);
		error=cl.Flush(commandqueue);	CHECKCL(error, clFlush);
		error=cl.Finish(commandqueue);	CHECKCL(error, clFinish);

		size_t worksize;

		worksize=9;
		error=cl.SetKernelArg(kernels[OCL_buf2tensor], 0, sizeof(cl_mem), &gpu_srcfilt);	CHECKCL(error, clSetKernelArg);
		error=cl.SetKernelArg(kernels[OCL_buf2tensor], 1, sizeof(cl_mem), &gpu_filt);	CHECKCL(error, clSetKernelArg);
		error=cl.EnqueueNDRangeKernel(commandqueue, kernels[OCL_buf2tensor], 1, 0, &worksize, 0, 0, 0, 0);	CHECKCL(error, clEnqueueNDRangeKernel);
		error=cl.Flush(commandqueue);	CHECKCL(error, clFlush);
		error=cl.Finish(commandqueue);	CHECKCL(error, clFinish);

		worksize=res;
		error=cl.SetKernelArg(kernels[OCL_image2tensor], 0, sizeof(cl_mem), &gpu_image);	CHECKCL(error, clSetKernelArg);
		error=cl.SetKernelArg(kernels[OCL_image2tensor], 1, sizeof(cl_mem), &gpu_dst);	CHECKCL(error, clSetKernelArg);
		error=cl.EnqueueNDRangeKernel(commandqueue, kernels[OCL_image2tensor], 1, 0, &worksize, 0, 0, 0, 0);	CHECKCL(error, clEnqueueNDRangeKernel);
		error=cl.Flush(commandqueue);	CHECKCL(error, clFlush);
		error=cl.Finish(commandqueue);	CHECKCL(error, clFinish);
		
		double elapsed=time_ms();
		long long cycles=__rdtsc();
		for(int it=0;it<100;++it)
		{
			cl_mem temp;
			SWAPVAR(gpu_src, gpu_dst, temp);

			worksize=res;
			error=cl.SetKernelArg(kernels[OCL_cc2d], 0, sizeof(cl_mem), &gpu_indices);	CHECKCL(error, clSetKernelArg);
			error=cl.SetKernelArg(kernels[OCL_cc2d], 1, sizeof(cl_mem), &gpu_filt);		CHECKCL(error, clSetKernelArg);
			error=cl.SetKernelArg(kernels[OCL_cc2d], 2, sizeof(cl_mem), &gpu_src);		CHECKCL(error, clSetKernelArg);
			error=cl.SetKernelArg(kernels[OCL_cc2d], 3, sizeof(cl_mem), &gpu_dst);		CHECKCL(error, clSetKernelArg);
			error=cl.EnqueueNDRangeKernel(commandqueue, kernels[OCL_cc2d], 1, 0, &worksize, 0, 0, 0, 0);	CHECKCL(error, clEnqueueNDRangeKernel);
			error=cl.Flush(commandqueue);	CHECKCL(error, clFlush);
			error=cl.Finish(commandqueue);	CHECKCL(error, clFinish);
		}
		worksize=res;
		error=cl.SetKernelArg(kernels[OCL_tensor2fp32], 0, sizeof(cl_mem), &gpu_dst);	CHECKCL(error, clSetKernelArg);
		error=cl.SetKernelArg(kernels[OCL_tensor2fp32], 1, sizeof(cl_mem), &gpu_image);	CHECKCL(error, clSetKernelArg);
		error=cl.EnqueueNDRangeKernel(commandqueue, kernels[OCL_tensor2fp32], 1, 0, &worksize, 0, 0, 0, 0);	CHECKCL(error, clEnqueueNDRangeKernel);
		error=cl.Flush(commandqueue);	CHECKCL(error, clFlush);
		error=cl.Finish(commandqueue);	CHECKCL(error, clFinish);

		error=cl.EnqueueReadBuffer(commandqueue, gpu_image, CL_TRUE, 0, res*sizeof(float), buf, 0, 0, 0);
		cycles=__rdtsc()-cycles;
		elapsed=time_ms()-elapsed;

		double sum=0;
		for(int k=0;k<res;++k)
		{
			float val=buf[k];
			sum+=(double)val*val;
		}
		sum=sqrt(sum);
		
		printf("%s\tOpenCL\t%12.6lfms  %12lldcycles  RMSE %14lf\n", testname, elapsed, cycles, sum);

		error=cl.ReleaseMemObject(gpu_indices);	CHECKCL(error, clReleaseMemObject);
		error=cl.ReleaseMemObject(gpu_filt);	CHECKCL(error, clReleaseMemObject);
		error=cl.ReleaseMemObject(gpu_src);	CHECKCL(error, clReleaseMemObject);
		error=cl.ReleaseMemObject(gpu_dst);	CHECKCL(error, clReleaseMemObject);

		cl.ReleaseProgram(program);
	}
	if(!halfprec_supported)
		printf("cl_khr_fp16 is NOT SUPPORTED\n");

	free(buf);
	error=cl.ReleaseCommandQueue(commandqueue);	CHECKCL(error, clReleaseCommandQueue);
	error=cl.ReleaseContext(context);		CHECKCL(error, clReleaseContext);
	return 0;
}
#endif