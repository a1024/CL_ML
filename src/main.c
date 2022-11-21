#include"acme_stdio.h"
#include"ocl_wrap.h"
#include"error.h"
#include"buffer.h"
#include"file.h"
#include"util.h"
#ifdef _MSC_VER
#include<Windows.h>
#endif
#include<stdarg.h>
#include<time.h>
#include<math.h>
#include<intrin.h>
#include"lodepng.h"
#define STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"
static const char file[]=__FILE__;

//	#define FIXED_PREC

//	#define DEBUG_SAVER
//	#define DEBUG_AUTOGRAD

#ifdef _MSC_VER
int set_console_buffer_size(short w, short h)
{
	COORD coords={w, h};
	HANDLE handle=GetStdHandle(STD_OUTPUT_HANDLE);
	int success=SetConsoleScreenBufferSize(handle, coords);
	if(!success)
		printf("Failed to resize console buffer: %d\n\n", GetLastError());
	return success;
}
#else
#define set_console_buffer_size(...)
#endif

int acme_print_memsize(char *buf, size_t bufsize, size_t memsize)
{
	if(memsize<1024)
		return sprintf_s(buf, bufsize, "%d Bytes", memsize);
	if(memsize<1024*1024)
		return sprintf_s(buf, bufsize, "%lf KB", (double)memsize*(1./1024));
	if(memsize<1024*1024*1024)
		return sprintf_s(buf, bufsize, "%lf MB", (double)memsize*(1./(1024*1024)));
	return sprintf_s(buf, bufsize, "%lf GB", (double)memsize*(1./(1024*1024*1024)));
}
int acme_strftime(char *buf, size_t size, double seconds)
{
	int hours=(int)floor(seconds/3600), minutes;
	seconds-=hours*3600;
	minutes=(int)floor(seconds/60);
	seconds-=minutes*60;
	return sprintf_s(buf, size, "%02d:%02d:%09.6lf", hours, minutes, seconds);
}
int acme_stricmp(const char *a, const char *b)
{
	if(!a||!b)
		return !a&&!b;
	while(*a&&tolower(*a)==tolower(*b))
		++a, ++b;
	return (*a>*b)-(*a<*b);
}
ptrdiff_t acme_strrchr(const char *str, ptrdiff_t len, char c)
{
	ptrdiff_t k;

	for(k=len-1;k>=0;--k)
		if(str[k]==c)
			return k;
	return -1;
}
const char* get_extension(const char *filename, ptrdiff_t len)
{
	ptrdiff_t idx;

	idx=acme_strrchr(filename, len, '.');
	if(idx==-1)
		return 0;
	return filename+idx+1;
#if 0
	const char *dot=strrchr(filename, '.');//https://stackoverflow.com/questions/5309471/getting-file-extension-in-c
	if(!dot||dot==filename)
		return "";
	return dot+1;
#endif
}
ArrayHandle filter_path(const char *path)
{
	ArrayHandle path2;
	char c;

	STR_COPY(path2, path, strlen(path));
	for(ptrdiff_t k=0;k<(ptrdiff_t)path2->count;++k)
	{
		if(path2->data[k]=='\\')
			path2->data[k]='/';
	}
	if(path2->data[path2->count-1]!='/')
	{
		c='/';
		STR_APPEND(path2, &c, 1, 1);
	}
	return path2;
}
void	free_str(void *p)
{
	ArrayHandle *str;
	
	str=(ArrayHandle*)p;
	array_free(str);
}
ArrayHandle get_filenames(const char *path, const char **extensions, int extCount)
{
	ArrayHandle searchpath, filename, filenames;
	char c;
	WIN32_FIND_DATAA data={0};
	void *hSearch;
	int success;
	const char *extension;
	ptrdiff_t len;
	int found;
	
	//prepare searchpath
	searchpath=filter_path(path);
	c='*';
	STR_APPEND(searchpath, &c, 1, 1);

	hSearch=FindFirstFileA(searchpath->data, &data);//skip .
	if(hSearch==INVALID_HANDLE_VALUE)
		return 0;
	success=FindNextFileA(hSearch, &data);//skip ..

	STR_POPBACK(searchpath, 1);//pop the '*'
	ARRAY_ALLOC(ArrayHandle, filenames, 0, 0, 0, free_str);

	for(;success=FindNextFileA(hSearch, &data);)
	{
		len=strlen(data.cFileName);
		extension=get_extension(data.cFileName, len);
		if(!(data.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY))
		{
			found=0;
			for(int k=0;k<extCount;++k)
			{
				if(!acme_stricmp(extension, extensions[k]))
				{
					found=1;
					break;
				}
			}
			if(found)
			{
				STR_COPY(filename, searchpath->data, searchpath->count);
				STR_APPEND(filename, data.cFileName, strlen(data.cFileName), 1);
				ARRAY_APPEND(filenames, &filename, 1, 1, 0);
			}
		}
	}
	success=FindClose(hSearch);
	array_free(&searchpath);
	return filenames;
}
void print_2d(const float *buf, int H, int W, int hex, const char *msg, ...)
{
	va_list args;
	if(msg)
	{
		va_start(args, msg);
		vprintf(msg, args);
		va_end(args);
		printf("\n");
	}
	const char *a;
	if(hex)
		a="%08X, ";
	else
		a="%f, ";
	for(int ky=0;ky<H;++ky)
	{
		for(int kx=0;kx<W;++kx)
			printf(a, buf[W*ky+kx]);
		printf("\n");
	}
	printf("\n");
}

#if 1
unsigned nplatforms=0, ndevices=0;
cl_platform_id *platforms=0;
cl_device_id *devices=0;
size_t maxlocalsize=0;
cl_context context=0;
cl_command_queue commandqueue=0;
cl_program program=0;

typedef enum		CLKernelIdxEnum
{
#define				CLKERNEL(LABEL)	OCL_##LABEL,
#include			"cl_kernel_names.h"
#undef				CLKERNEL
	OCL_NKERNELS,
} CLKernelIdx;
const char			*kernelnames[]=
{
#define				CLKERNEL(LABEL)	#LABEL,
#include			"cl_kernel_names.h"
#undef				CLKERNEL
#ifdef __INTELLISENSE__
	0,
#endif
};
cl_kernel			kernels[OCL_NKERNELS]={0};
void				ocl_init(const char *srcname)
{
	int error;

	oclw_loadAPI();

	error=p_clGetPlatformIDs(0, 0, &nplatforms);		CL_CHECK(error);
	ASSERT_MSG(nplatforms, "No OpenCL platforms");
	
	platforms=(cl_platform_id*)malloc(nplatforms*sizeof(cl_platform_id));
	error=p_clGetPlatformIDs(nplatforms, platforms, 0);	CL_CHECK(error);

	error=p_clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, 0, &ndevices);	CL_CHECK(error);
	ASSERT_MSG(ndevices, "No OpenCL devices");

	devices=(cl_device_id*)malloc(ndevices*sizeof(cl_device_id));
	error=p_clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, ndevices, devices, 0);	CL_CHECK(error);
	
	//get info
	size_t retlen=0;
	error=p_clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, G_BUF_SIZE, g_buf, &retlen);CL_CHECK(error);
	printf("OpenCL platform: %s\n", g_buf);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_NAME, G_BUF_SIZE, g_buf, &retlen);	CL_CHECK(error);
	printf("Device: %s\n", g_buf);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, G_BUF_SIZE, g_buf, &retlen);	CL_CHECK(error);
	printf("Vendor: %s\n", g_buf);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxlocalsize, &retlen);	CL_CHECK(error);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE = %d\n", (int)maxlocalsize);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_ADDRESS_BITS, sizeof(size_t), &maxlocalsize, &retlen);	CL_CHECK(error);
	printf("CL_DEVICE_ADDRESS_BITS = %d\n", (int)maxlocalsize);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, G_BUF_SIZE, g_buf, &retlen);	CL_CHECK(error);
	for(int k=0;k<retlen;++k)
		if(g_buf[k]==' ')
			g_buf[k]='\n';
	printf("Extensions:\n%s\n", g_buf);
	//int cl_gl_interop=strstr(g_buf, "cl_khr_gl_sharing")!=0;
	//if(!cl_gl_interop)
	//	printf("\n\tcl_khr_gl_sharing not supported\n\n");
	
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
	context=p_clCreateContext(properties, 1, &devices[0], nullptr, nullptr, &error);	CL_CHECK(error);
#else
	context=p_clCreateContext(0, ndevices, devices, 0, 0, &error);			CL_CHECK(error);
#endif
	commandqueue=p_clCreateCommandQueue(context, devices[0], 0, &error);	CL_CHECK(error);

	//compile kernel
	ArrayHandle srctext=load_text(srcname, 0);
	ASSERT_MSG(srctext, "Cannot open \'%s\'", srcname);
	const char *k_src=(const char*)srctext->data;
	size_t k_len=srctext->count;
	program=p_clCreateProgramWithSource(context, 1, (const char**)&k_src, &k_len, &error);	CL_CHECK(error);
	const char defines[]="-D__OPEN_CL__"
#ifdef FIXED_PREC
		" -DFIXED_PREC"
#endif
		;
	error=p_clBuildProgram(program, 1, devices, defines, 0, 0);
	if(error)
	{
		error=p_clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, G_BUF_SIZE, g_buf, &retlen);
		if(retlen>G_BUF_SIZE)
		{
			char *buf=(char*)malloc(retlen+10);
			error=p_clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, retlen+10, buf, &retlen);	CL_CHECK(error);
			printf("\nOpenCL Compilation failed:\n%s\n", buf);
			free(buf);
			LOG_ERROR("Aborting");
		}
		else
			LOG_ERROR("\nOpenCL Compilation failed:\n%s\n", g_buf);
	}

	for(int k=0;k<OCL_NKERNELS;++k)
	{
		kernels[k]=p_clCreateKernel(program, kernelnames[k], &error);
		ASSERT_MSG(!error, "Couldn't find kernel %s", kernelnames[k]);
	}
}
#endif

#if 1
void init_buf(double *buf, int size, double mag)
{
	mag/=RAND_MAX;
	for(int k=0;k<size;++k)
		buf[k]=mag*rand();
}
void add_sh1d(const double *src, double gain, double *dst, int N, int sh)//shifted addition, sh: src forward
{
	for(int k=0;k<N;++k)
	{
		unsigned idx=k+sh;
		if(idx<(unsigned)N)
			dst[idx]+=src[k]*gain;
	}
}
void emul_sh1d(const double *x1, const double *x2, double *dst, int N, int sh)//shifted addition, sh: src forward	UNUSED
{
	for(int k=0;k<N;++k)
	{
		unsigned idx=k+sh;
		if(idx<(unsigned)N)
			dst[idx]=x1[idx]*x2[k];
	}
}
double dot_sh1d(const double *x1, const double *x2, int N, int sh)//shifted addition, sh: src forward
{
	double sum=0;
	for(int k=0;k<N;++k)
	{
		unsigned idx=k+sh;
		if(idx<(unsigned)N)
			sum+=x1[idx]*x2[k];
	}
	return sum;
}
void subtract(const double *pos, const double *neg, double *dst, int N)
{
	for(int k=0;k<N;++k)
		dst[k]=pos[k]-neg[k];
}
double sum_1d(const double *x, int N)
{
	double sum=0;
	for(int k=0;k<N;++k)
		sum+=x[k];
	return sum;
}
double magsq(const double *x, int N)
{
	double sum=0;
	for(int k=0;k<N;++k)
		sum+=x[k]*x[k];
	return sum;
}
void emul(const double *x1, const double *x2, double *dst, int N)
{
	for(int k=0;k<N;++k)
		dst[k]=x1[k]*x2[k];
}
void gmul(const double *src, double gain, double *dst, int N)
{
	for(int k=0;k<N;++k)
		dst[k]=src[k]*gain;
}
void cc1d(const double *x, const double *kernel, double bias, double *dst, int N, int K)//cross-correlation, zero-pad=(K-1)/2
{
	int P=(K-1)>>1;
	for(int k=0;k<N;++k)
		dst[k]=bias;
	for(int k=0;k<K;++k)
		add_sh1d(x, kernel[k], dst, N, k-P);
}
void conv1d(const double *x, const double *kernel, double bias, double *dst, int N, int K)//cross-correlation, zero-pad=(K-1)/2
{
	int P=(K-1)>>1;
	for(int k=0;k<N;++k)
		dst[k]=bias;
	for(int k=0;k<K;++k)
		add_sh1d(x, kernel[k], dst, N, (K-1-k)-P);
}
void LeakyReLU(const double *src, double *dst, int N)
{
	for(int k=0;k<N;++k)
	{
		double x2=0.01*src[k];
		dst[k]=x2>src[k]?x2:src[k];
	}
}
void LeakyReLU_d(const double *src, double *dst, int N)
{
	for(int k=0;k<N;++k)
		dst[k]=src[k]<0?-0.01:1;
}
void ReLU(const double *src, double *dst, int N)
{
	for(int k=0;k<N;++k)
		dst[k]=src[k]<0?0:src[k];
}
void ReLU_d(const double *src, double *dst, int N)
{
	for(int k=0;k<N;++k)
		dst[k]=src[k]<0?0:1;
}
void mix_inplace(double *dst, const double *src, double x, size_t N)//dst = x*dst + (1-x)*src
{
	for(ptrdiff_t k=0;k<(ptrdiff_t)N;++k)
		dst[k]=src[k]+(dst[k]-src[k])*x;
}
void sq_inplace(double *x, size_t N)
{
	for(ptrdiff_t k=0;k<(ptrdiff_t)N;++k)
		x[k]*=x[k];
}
void print_1d(const double *x, int N, const char *msg, ...)
{
	va_list args;
	if(msg)
	{
		va_start(args, msg);
		vprintf(msg, args);
		va_end(args);
	}
	for(int k=0;k<N;++k)
		printf("%lf, ", x[k]);
	printf("\n");
}
#endif

//example hardcoded 1D convnet
#if 0
#define sizeK	3
#define sizeN	4
#define nLayers	3
#define nParams	(sizeK+1)*nLayers
double
	x[]={1, 2, 3, 4},

	params[nParams]=
	{
		0, 1, 0,	0,
		0, 1, 0,	0,
		0, 1, 0,	0,
	},
	gradient[nParams],

	*filt1=params+0, *bias1=params+3,
	*filt2=params+4, *bias2=params+7,
	*filt3=params+8, *bias3=params+11,

	net1[sizeN], t1[sizeN], net2[sizeN], t2[sizeN], net3[sizeN], xh[sizeN],
	diff[sizeN],

	dL_dnet3[sizeN], dL_dt2[sizeN],		*dL_dfilt3=gradient+8, *dL_dbias3=gradient+11,
	dL_dnet2[sizeN], dL_dt1[sizeN],		*dL_dfilt2=gradient+4, *dL_dbias2=gradient+7,
	dL_dnet1[sizeN], dL_dx[sizeN],		*dL_dfilt1=gradient+0, *dL_dbias1=gradient+3;
double adam_m[nParams]={0}, adam_v[nParams]={0};
#endif


//automatic differentiation in 2D
#if 1
int using_gpu=0;
typedef enum AugmentationTypeEnum
{
	AUG_BLOCK,		//image is padded then partitioned to HxW blocks
	AUG_RANDCROP,	//random crop
	AUG_STRETCH,	//image is stretched to a HxW block
} AugmentationType;
typedef enum InstTypeEnum
{
	//FWD: x[C,H,W], filt[Co,Ci,K,K], bias[Co] -> net[C,H,W]
	//BWD: dL_dnet[C,H,W], filt[Co,Ci,K,K], x[C,H,W] -> dL_dx[C,H,W], dL_dfilt[Co,Ci,K,K], dL_dbias[Co]
	OP_CC2,
	OP_CONV2,

	//FWD: net[C,H,W] -> x2[C,H,W]
	//BWD: dL_dx2[C,H,W], net[C,H,W] -> dL_dnet[C,H,W]
	OP_LRELU,
	OP_RELU,

	//FWD: x[C,H,W] -> xhat=clamp(x[C,H,W] + qnoise[C,H,W]), replaced with real quantizer at test time
	//BWD: dL_dxhat[C,H,W] -> dL_dx=dL_dxhat (identity)
	OP_QUANTIZER,

	//FWD: yhat[C,H,W], y[C,H,W] -> diff[C,H,W], L[1]
	//BWD: (void) -> dL_dyhat=diff
	OP_MSE,
	OP_MS_SSIM,
} InstType;
//typedef enum OperationTypeEnum
//{
//	OP_CROSSCORR,
//	OP_CONV,
//} OperationType;
//typedef enum NLTypeEnum
//{
//	NL_LEAKYRELU,
//	NL_RELU,
//} NLType;
typedef struct InstructionStruct
{
	InstType op;
	int fwd_args[3], fwd_result, bwd_args[3], bwd_results[3],//indices
		info[4],//[xpad, ypad, xstride, ystride], or [quantization_nlevels]
		save_result;
} Instruction;
typedef enum BufferTypeEnum
{
	//use buf->data
	BUF_NORMAL, BUF_NORMAL_SAVED,

	//use model->params->data+buf->offset*sizeof(double)
	BUF_PARAM, BUF_GRAD,
} BufferType;
typedef struct BufferStruct
{
	//[B, C, H, W]				for data & its gradient
	//[Cout, Cin, Ky, Kx]		for conv filter & its gradient (in-place added-up over B)
	//[Cout, 1, 1, 1]			for conv bias & its gradient (in-place added-up over B)
	int shape[4];
	BufferType type;
	union
	{
		size_t offset;
		double *data;
		cl_mem gpu_buf;
	};

	//size_t param_offset,//		if !data:			buffer is a learnable parameter, use model->params->data+param_offset*sizeof(double)
	//	grad_offset;	//else	if (size_t)data<16: buffer is a gradient parameter,  use model->params->data+grad_offset*sizeof(double)
	//double *data;		//else:						use data
} Buffer;
typedef struct ModelStruct
{
	AugmentationType aug;
	int input_shape[4],//[B,C,W,H]
		is_test,
		input;//index in buffers
	ArrayHandle
		src,				//string with the model source code, for saving
		testpath, trainpath,//strings with paths to datasets
		instructions,	//array of Instruction
		buffers;		//array of Buffer/cl_mem, depending on using_gpu
	size_t nparams;
	union
	{
		struct
		{
			ArrayHandle cpu_params, cpu_grad, cpu_adam_m, cpu_adam_v;//arrays of double (same size)
		};
		struct
		{
			cl_mem gpu_params, gpu_grad, gpu_adam_m, gpu_adam_v;
		};
	};
	float *gpu_input, *gpu_output;
} Model;
void free_buffer(void *p)
{
	Buffer *buf=(Buffer*)p;
	if(buf->type==BUF_NORMAL||buf->type==BUF_NORMAL_SAVED)
	{
		if(using_gpu)
		{
			int error=p_clReleaseMemObject(buf->gpu_buf);
			CL_CHECK(error);
			buf->gpu_buf=0;
		}
		else
			free(buf->data);
	}
}
void free_model(void *p)
{
	Model *model=(Model*)p;
	array_free(&model->src);
	array_free(&model->instructions);
	array_free(&model->buffers);
	if(using_gpu)
	{
		int error;
		error=p_clReleaseMemObject(model->gpu_params);	CL_CHECK(error);	model->gpu_params=0;
		error=p_clReleaseMemObject(model->gpu_grad);	CL_CHECK(error);	model->gpu_grad=0;
		error=p_clReleaseMemObject(model->gpu_adam_m);	CL_CHECK(error);	model->gpu_adam_m=0;
		error=p_clReleaseMemObject(model->gpu_adam_v);	CL_CHECK(error);	model->gpu_adam_v=0;
	}
	else
	{
		array_free(&model->cpu_adam_m);
		array_free(&model->cpu_adam_v);
		array_free(&model->cpu_grad);
		array_free(&model->cpu_params);
	}
}
typedef struct VariableStruct
{
	ArrayHandle name;
	int buffer;
} Variable;
void free_variable(void *p)
{
	Variable *var=(Variable*)p;
	array_free(&var->name);
}

void preview_cl_buffer(Buffer *buf, cl_mem clbuf, size_t offset, size_t bufsize)
{
	size_t size=bufsize?bufsize:buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
	float *buffer=(float*)malloc(size*sizeof(float));
	int error=p_clEnqueueReadBuffer(commandqueue, clbuf?clbuf:buf->gpu_buf, CL_TRUE, offset, size*sizeof(float), buffer, 0, 0, 0);	CL_CHECK(error);
	if(bufsize)
	{
		for(int k=0;k<bufsize;++k)
			printf("%4d %g\n", k, buffer[k]);
	}
	else
	{
		int dx=10, dy=50, dc=2;
		if(dx>buf->shape[3])
			dx=buf->shape[3];
		if(dy>buf->shape[2])
			dy=buf->shape[2];
		if(dc>buf->shape[1])
			dc=buf->shape[1];
		for(int kc=0;kc<dc;++kc)
		{
			for(int ky=0;ky<dy;++ky)
			{
				for(int kx=0;kx<dx;++kx)
					printf("%g ", buffer[buf->shape[3]*(buf->shape[2]*kc+ky)+kx]);
				printf("\n");
			}
			printf("\n");
		}
	}
	printf("\n");
	for(int k=0;k<size;++k)
	{
		if(!isfinite(buffer[k])||fabsf(buffer[k])>1e9)
		{
			printf("%d: %g\n", k, buffer[k]);
			break;
		}
	}
}

//parser
#define			DUPLET(A, B)	((unsigned char)(A)|(unsigned char)(B)<<8)
int				skip_ws(ArrayHandle text, int *ctx)
{
	int *idx=ctx, *lineno=ctx+1, *linestart=ctx+2;
	for(;*idx<(int)text->count;)
	{
		if(text->data[*idx]=='/')
		{
			if(text->data[*idx+1]=='/')//line comment
			{
				for(;*idx<(int)text->count&&text->data[*idx]&&text->data[*idx]!='\n';++*idx);
				int inc=*idx<(int)text->count;
				*idx+=inc;//skip newline
				*lineno+=inc;
				*linestart=*idx;
			}
			else if(text->data[*idx+1]=='*')//block comment
			{
				for(unsigned short data=DUPLET(text->data[*idx], text->data[*idx+1]);;)
				{
					if(*idx>=(int)text->count)
					{
						//lex_error(p, len, k, "Expected \'*/\'");
						break;
					}
					if(data==DUPLET('*', '/'))
					{
						*idx+=2;
						break;
					}
					if((data&0xFF)=='\n')
					{
						++*idx;
						++*lineno;
						*linestart=*idx;
					}
					else
						++*idx;
					data>>=8, data|=(unsigned char)text->data[*idx+1]<<8;
				}
			}
			else
				break;
		}
		else if(isspace(text->data[*idx]))
		{
			if(text->data[*idx]=='\n')
			{
				++*idx;
				++*lineno;
				*linestart=*idx;
			}
			else
				++*idx;
		}
		else
			break;
	}
	return *idx>=(int)text->count;
}
int				match_kw(ArrayHandle text, int *idx, const char **keywords, int nkw)//returns the index of matched keyword
{
	for(int kk=0;kk<nkw;++kk)
	{
		const char *kw=keywords[kk];
		int k=*idx, k2=0;
		for(;k<(int)text->count;++k, ++k2)
		{
			if(!kw[k2])//match
			{
				*idx=k;
				return kk;
			}
			if(text->data[k]!=kw[k2])
				break;
		}
	}
	return -1;
}
int				get_id(ArrayHandle text, int *idx)//returns true if valid identifier
{
	int valid=isalpha(text->data[*idx])||text->data[*idx]=='_';
	if(!valid)
		return 0;
	do
		++*idx;
	while(isalnum(text->data[*idx])||text->data[*idx]=='_');
	return 1;
}
int				parse_number(const char *filename, ArrayHandle text, int *ctx, int base, double *ret_val)//units_type: 0: no units, 1: cm, 2: nm
{
	int *idx=ctx, *lineno=ctx+1, *linestart=ctx+2;
	double val=0;
	int success=0;
	double sign;

	if(text->data[*idx]=='-')
	{
		++*idx;//skip minus
		sign=-1;
		if(skip_ws(text, ctx))
		{
			LOG_ERROR("%s(%d): Expected a number", filename, *lineno);
			return 0;
		}
	}
	else
		sign=1;
	for(;*idx<(int)text->count;++*idx)
	{
		unsigned char c=text->data[*idx]-'0';
		if(c>=10)
		{
			if(base!=16)
				break;
			c=(text->data[*idx]&0xDF)-'A';
			if(c>=6)
				break;
			c+=10;
		}
		val*=base;
		val+=c;
		success=1;
	}
	if(text->data[*idx]=='.')
	{
		++*idx;//skip point
		double p=1, dp=1./base;
		for(;*idx<(int)text->count;++*idx)
		{
			unsigned char c=text->data[*idx]-'0';
			if(c>=10)
			{
				if(base!=16)
					break;
				c=(text->data[*idx]&0xDF)-'A';
				if(c>=6)
					break;
				c+=10;
			}
			p*=dp;
			val+=c*p;
			success=1;
		}
	}
	if(!success)
		return -1;
	if(ret_val)
		*ret_val=sign*val;
	return success;
}
#if 0
void			skip_until(ArrayHandle text, int *ctx, char c)
{
	for(;text->data[*ctx]&&text->data[*ctx]!=c;++*ctx)
	{
		if(text->data[*ctx]=='\n')
		{
			++ctx[1];
			ctx[2]=*ctx;
		}
	}
}
#endif
ArrayHandle		get_path(const char *filename, ArrayHandle text, int *ctx)
{
	ArrayHandle ret;
	size_t start, end;
	char c=text->data[*ctx];
	if(c=='\''||c=='\"')
	{
		++*ctx;//skip opening quote
		start=*ctx;
		for(;*ctx<text->count&&text->data[*ctx]!=c;++*ctx)
		{
			if(text->data[*ctx]=='\n')
			{
				++ctx[1];
				ctx[2]=*ctx;
				LOG_ERROR("%s(%d): Missing closing \'%c\'. Path cannot contain newlines", filename, ctx[1], c);
				break;
			}
		}
		end=*ctx;
		++*ctx;//skip closing quote
	}
	else
	{
		start=*ctx;
		for(;text->data[*ctx]&&!isspace(text->data[*ctx]);++*ctx);
		end=*ctx;
	}
	STR_COPY(ret, text->data+start, end-start);
	for(int k=0;k<ret->count;++k)
	{
		if(ret->data[k]=='\\')
			ret->data[k]='/';
	}
	if(ret->data[ret->count-1]=='/')
	{
		ret->data[ret->count-1]='\0';
		--ret->count;
	}
	int result=file_is_readable((char*)ret->data);
	ASSERT_MSG(result, "Path doesn't exist: \'%s\'", ret->data);
	ASSERT_MSG(result==2, "Path is a file, expected a folder: \'%s\'", ret->data);
	STR_APPEND(ret, "/", 1, 1);
	return ret;
}

const char
	kw_train[]="train", kw_test[]="test",
	kw_block[]="block", kw_randcrop[]="randcrop", kw_stretch[]="stretch",
	kw_save[]="save",
	kw_cc[]="cc", 
	kw_conv[]="conv",
		kw_lrelu[]="lrelu", kw_relu[]="relu",
	kw_quantize[]="quantize",
	kw_loss[]="loss",
		kw_mse[]="mse",
		kw_msssim[]="ms-ssim",
	kw_weights[]="weights";
const char
	*ksearch_dataset[]={kw_train, kw_test},
	*ksearch_aug[]={kw_block, kw_randcrop, kw_stretch},
	*ksearch_global[]={kw_cc, kw_conv, kw_save, kw_quantize, kw_loss},
	*ksearch_nl[]={kw_lrelu, kw_relu},
	*ksearch_loss[]={kw_mse, kw_msssim};

void parse_conv(const char *filename, ArrayHandle text, int *ctx, Model* model, ArrayHandle variables, int match)
{
	Instruction *inst;
	Buffer *buffer;
	double fval;
	int ival;

	int first_inst=!model->instructions->count;
	inst=(Instruction*)ARRAY_APPEND(model->instructions, 0, 2, 1, 0);
	inst->op=match?OP_CONV2:OP_CC2;

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	ival=(int)round(fval);
	ASSERT_MSG(ival==2, "%s(%d): Only 2D operations are supported, nDim = ", filename, ctx[1], ival);

	int Cin, Cout, K, stride, pad;
	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	Cin=(int)round(fval);

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	Cout=(int)round(fval);

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	K=(int)round(fval);
	
	skip_ws(text, ctx);
	if(text->data[*ctx]=='x')
	{
		++*ctx;
		skip_ws(text, ctx);
		parse_number(filename, text, ctx, 10, &fval);
		ival=(int)round(fval);
		ASSERT_MSG(K==ival, "%s(%d): Only square filters are supported, filter=%dx%d", filename, ctx[1], K, ival);
	}

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	stride=(int)round(fval);
	//ASSERT_MSG(stride==1, "%s(%d): Only stride of 1 is supported, stride=%d", filename, ctx[1], stride);

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	pad=(int)round(fval);
	ASSERT_MSG((pad<<1)==K-1, "%s(%d): Only padding of (K-1)/2 is supported, K=%d, pad=%d", filename, ctx[1], K, pad);

	skip_ws(text, ctx);
	match=match_kw(text, ctx, ksearch_nl, COUNTOF(ksearch_nl));
	switch(match)
	{
	case -1:
		LOG_ERROR("%s(%d): Expected \'lrelu\' or \'relu\'", filename, ctx[1]);
		break;
	case 0:
		inst[1].op=OP_LRELU;
		break;
	case 1:
		inst[1].op=OP_RELU;
		break;
	}

	int nbuffers=(int)model->buffers->count;
	buffer=(Buffer*)ARRAY_APPEND(model->buffers, 0, 8+(first_inst<<1), 1, 0);
	int x_idx, dLdx_idx;
	if(first_inst)
	{
		x_idx=nbuffers+8;
		dLdx_idx=nbuffers+9;
	}
	else
	{
		x_idx=inst[-1].fwd_result;
		dLdx_idx=inst[-1].bwd_args[0];
	}
	Buffer
		*x=(Buffer*)array_at(&model->buffers, x_idx),
		*dL_dx=(Buffer*)array_at(&model->buffers, dLdx_idx),
		*filt=buffer, *dL_dfilt=buffer+1, *bias=buffer+2, *dL_dbias=buffer+3, *net=buffer+4, *dL_dnet=buffer+5,
		*x2=buffer+6, *dL_dx2=buffer+7;
	if(first_inst)
	{
		x->shape[0]=model->input_shape[0];//B
		x->shape[1]=model->input_shape[1]=Cin;
		x->shape[2]=model->input_shape[2];
		x->shape[3]=model->input_shape[3];
		x->type=BUF_NORMAL;
		memcpy(dL_dx->shape, x->shape, sizeof(x->shape));
		dL_dx->type=BUF_NORMAL;
		for(int k=0;k<(int)variables->count;++k)
		{
			Variable *var=(Variable*)array_at(&variables, k);
			if(var->buffer==-1)
				var->buffer=x_idx;
		}
	}
	filt->shape[0]=Cout;
	filt->shape[1]=Cin;
	filt->shape[2]=K;
	filt->shape[3]=K;
	filt->type=BUF_PARAM;
	memcpy(dL_dfilt->shape, filt->shape, sizeof(filt->shape));
	dL_dfilt->type=BUF_GRAD;//gradient
	bias->shape[0]=Cout;
	bias->shape[1]=1;
	bias->shape[2]=1;
	bias->shape[3]=1;
	bias->type=BUF_PARAM;//param
	memcpy(dL_dbias->shape, bias->shape, sizeof(bias->shape));
	dL_dbias->type=BUF_GRAD;//gradient
	net->shape[0]=model->input_shape[0];
	net->shape[1]=Cout;
	net->shape[2]=(x->shape[2]+(pad<<1)-(K-1)-1)/stride+1;
	net->shape[3]=(x->shape[3]+(pad<<1)-(K-1)-1)/stride+1;
	net->type=BUF_NORMAL;
	memcpy(dL_dnet->shape, net->shape, sizeof(net->shape));
	dL_dnet->type=BUF_NORMAL;
	memcpy(x2->shape, net->shape, sizeof(net->shape));
	x2->type=BUF_NORMAL;
	memcpy(dL_dx2->shape, net->shape, sizeof(net->shape));
	dL_dx2->type=BUF_NORMAL;

	//cc/conv
	inst->fwd_args[0]=x_idx;		//x
	inst->fwd_args[1]=nbuffers;		//filt
	inst->fwd_args[2]=nbuffers+2;	//bias
	inst->fwd_result=nbuffers+4;	//net
	inst->bwd_args[0]=nbuffers+5;	//dL_dnet
	inst->bwd_args[1]=nbuffers;		//filt
	inst->bwd_args[2]=x_idx;
	inst->bwd_results[0]=dLdx_idx;	//dL_dx
	inst->bwd_results[1]=nbuffers+1;//dL_dfilt
	inst->bwd_results[2]=nbuffers+3;//dL_dbias
	inst->info[0]=inst->info[1]=pad;	//[x,y]pad
	inst->info[2]=inst->info[3]=stride;	//[x,y]stride

	//nonlinearity
	inst[1].fwd_args[0]=nbuffers+4;		//net
	inst[1].fwd_result=nbuffers+6;		//x2
	inst[1].bwd_args[0]=nbuffers+7;		//dL_dx2
	inst[1].bwd_args[1]=nbuffers+6;		//x2
	inst[1].bwd_results[0]=nbuffers+5;	//dL_dnet
}
void parse_save(const char *filename, ArrayHandle text, int *ctx, Model* model, ArrayHandle *variables)
{
	Variable *var;
	int start;
	Instruction *inst;

	skip_ws(text, ctx);
	start=*ctx;
	get_id(text, ctx);
	int found=0;
	for(int k=0;k<(int)variables[0]->count;++k)
	{
		var=(Variable*)array_at(variables, k);
		if(var->name->count==*ctx-start&&!memcmp(text->data+start, var->name->data, var->name->count))
		{
			found=1;
			break;
		}
	}
	ASSERT_MSG(!found, "%s(%d): Saved variable can only appear once, \'%.*s\' appeared before", filename, ctx[1], *ctx-start, text->data+start);
	var=(Variable*)ARRAY_APPEND(*variables, 0, 1, 1, 0);
	STR_COPY(var->name, text->data+start, *ctx-start);
	if(model->instructions->count)
	{
		inst=(Instruction*)array_at(&model->instructions, model->instructions->count-1);
		var->buffer=inst->fwd_result;
	}
	else
		var->buffer=-1;
}
void parse_quantize(const char *filename, ArrayHandle text, int *ctx, Model* model, ArrayHandle variables)
{
	Instruction *inst;
	Buffer *buffer;
	double fval;
	
	ASSERT_MSG(model->instructions->count, "%s(%d): Quantizer cannot be the first operation", filename, *ctx);//because Cin would be unknown
	int first_inst=!model->instructions->count;
	inst=(Instruction*)ARRAY_APPEND(model->instructions, 0, 1, 1, 0);
	inst->op=OP_QUANTIZER;

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	inst->info[0]=(int)round(fval);//nlevels
	ASSERT_MSG(inst->info[0]>1, "Quantization nlevels should be greater than 1, got %d", inst->info[0]);

	int nbuffers=(int)model->buffers->count;
	buffer=(Buffer*)ARRAY_APPEND(model->buffers, 0, 1+(first_inst<<1), 1, 0);
	int x_idx, dLdx_idx;
	if(first_inst)
	{
		x_idx=nbuffers+1;
		dLdx_idx=nbuffers+2;
	}
	else
	{
		x_idx=inst[-1].fwd_result;
		dLdx_idx=inst[-1].bwd_args[0];
	}
	Buffer
		*x=(Buffer*)array_at(&model->buffers, x_idx),
		*dL_dx=(Buffer*)array_at(&model->buffers, dLdx_idx),
		*xhat=buffer;
	if(first_inst)
	{
		x->shape[0]=model->input_shape[0];//B
		x->shape[1]=model->input_shape[1];//FIXME: Cin is unknown if quantizer appears first
		x->shape[2]=model->input_shape[2];
		x->shape[3]=model->input_shape[3];
		x->type=BUF_NORMAL;//not a param
		memcpy(dL_dx->shape, x->shape, sizeof(x->shape));
		dL_dx->type=BUF_NORMAL;
		for(int k=0;k<(int)variables->count;++k)
		{
			Variable *var=(Variable*)array_at(&variables, k);
			if(var->buffer==-1)
				var->buffer=x_idx;
		}
	}
	memcpy(xhat->shape, x->shape, sizeof(x->shape));
	xhat->type=BUF_NORMAL;

	inst->fwd_args[0]=x_idx;
	inst->fwd_result=nbuffers;
	inst->bwd_args[0]=dLdx_idx;//should be dL_dxhat, but quantizer.bwd is identity
	inst->bwd_results[0]=dLdx_idx;
}
int match_varname(const char *varname, int len, ArrayHandle variables)
{
	for(int k=0;k<(int)variables->count;++k)
	{
		Variable *v2=array_at(&variables, k);
		if(len==v2->name->count&&!memcmp(varname, v2->name->data, len))
			return v2->buffer;
	}
	return -1;
}
void parse_loss(const char *filename, ArrayHandle text, int *ctx, Model* model, ArrayHandle variables)
{
	Instruction *inst;
	Buffer *buffer;
	int start;
	
	ASSERT_MSG(model->instructions->count, "%s(%d): Loss cannot be the first operation", filename, *ctx);
	inst=(Instruction*)ARRAY_APPEND(model->instructions, 0, 1, 1, 0);
	
	skip_ws(text, ctx);
	int match=match_kw(text, ctx, ksearch_loss, COUNTOF(ksearch_loss));
	switch(match)
	{
	case -1:
		LOG_ERROR("%s(%d): Expected \'mse\' or \'ms-ssim\'", filename, ctx[1]);
		break;
	case 0:
		inst->op=OP_MSE;
		break;
	case 1:
		inst->op=OP_MS_SSIM;
		break;
	}
	
	skip_ws(text, ctx);
	start=*ctx;
	get_id(text, ctx);
	int b1=match_varname(text->data+start, *ctx-start, variables);
	ASSERT_MSG(b1!=-1, "%s(%d): Undeclared identifier \'%.*s\'", filename, ctx[1], *ctx-start, text->data+start);

	skip_ws(text, ctx);
	start=*ctx;
	get_id(text, ctx);
	int b2=match_varname(text->data+start, *ctx-start, variables);
	ASSERT_MSG(b2!=-1, "%s(%d): Undeclared identifier \'%.*s\'", filename, ctx[1], *ctx-start, text->data+start);
	
	int nbuffers=(int)model->buffers->count;
	buffer=(Buffer*)ARRAY_APPEND(model->buffers, 0, 1, 1, 0);//diff

	Buffer
		*buf1=(Buffer*)array_at(&model->buffers, b1),
		*buf2=(Buffer*)array_at(&model->buffers, b2),
		*diff=buffer;
	ASSERT_MSG(!memcmp(buf1->shape, buf2->shape, sizeof(buf2->shape)),
		"%s(%d): Loss function expects two buffers of identical shape, got [%d %d %d %d] and [%d %d %d %d]", filename, ctx[1],
		buf1->shape[0], buf1->shape[1], buf1->shape[2], buf1->shape[3], buf2->shape[0], buf2->shape[1], buf2->shape[2], buf2->shape[3]);
	memcpy(diff->shape, buf1->shape, sizeof(buf1->shape));
	diff->type=BUF_NORMAL;

	inst->fwd_args[0]=b1;
	inst->fwd_args[1]=b2;
	inst->fwd_result=nbuffers;
	inst->bwd_results[0]=nbuffers;
}
void init_model(Model* model)
{
	//for(int ki=0;ki<(int)model->instructions->count;++ki)//Xavier initialization
	//{
	//	Instruction *inst=(Instruction*)array_at(&model->instructions, ki);
	//}
	double gain=1./RAND_MAX;
	size_t kp=0;
	ArrayHandle temp=0;
	int error;
	if(using_gpu)
		ARRAY_ALLOC(float, temp, 0, model->nparams, 0, 0);
	for(int kb=0;kb<(int)model->buffers->count;++kb)
	{
		Buffer *buf=(Buffer*)array_at(&model->buffers, kb);
		if(buf->type==BUF_PARAM)
		{
			size_t bufsize=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
			double g2=gain*sqrt(6./(buf->shape[0]+buf->shape[1]));
			if(using_gpu)
			{
				for(size_t k0=0;k0<bufsize;++k0, ++kp)
				{
					float *val=(float*)array_at(&temp, kp);
					*val=(float)((rand()-(RAND_MAX>>1))*g2);//[-0.5, 0.5]
#ifdef FIXED_PREC
					*(int*)val=*val*0x10000;
#endif
					//printf("%4d: %08X %g\n", kp, *(int*)val, *val);
				}
			}
			else
			{
				for(size_t k0=0;k0<bufsize;++k0, ++kp)
				{
					double *val=(double*)array_at(&model->cpu_params, kp);
					*val=(rand()-(RAND_MAX>>1))*g2;//[-0.5, 0.5]
				}
			}
		}
	}
	if(using_gpu)
	{
		error=p_clEnqueueWriteBuffer(commandqueue, model->gpu_params, CL_TRUE, 0, model->nparams*sizeof(float), temp->data, 0, 0, 0);	CL_CHECK(error);
		array_free(&temp);
#if 0
		preview_cl_buffer(0, model->gpu_params, 0, model->nparams);//
#endif
	}
}
size_t parse_model(const char *filename, Model* model)
{
	ArrayHandle text=load_text(filename, 0),
		variables;//array of Variable
	int ctx[3]={0}, match;//idx, lineno, linestart
	double fval=0;
	int ival;
	if(!text)
		LOG_ERROR("Cannot open \'%s\'\n", filename);

	model->trainpath=0;
	model->testpath=0;
	for(int k=0;k<2;++k)
	{
		skip_ws(text, ctx);
		match=match_kw(text, ctx, ksearch_dataset, COUNTOF(ksearch_dataset));
		switch(match)
		{
		case -1:
			LOG_ERROR("%s(%d): Expected \'train\' or \'test\'", filename, ctx[1]);
		case 0:
			ASSERT_MSG(!model->trainpath, "Train path already encountered");
			skip_ws(text, ctx);
			model->trainpath=get_path(filename, text, ctx);
			break;
		case 1:
			ASSERT_MSG(!model->testpath, "Test path already encountered");
			skip_ws(text, ctx);
			model->testpath=get_path(filename, text, ctx);
			break;
		}
	}

	skip_ws(text, ctx);
	match=match_kw(text, ctx, ksearch_aug, COUNTOF(ksearch_aug));
	switch(match)
	{
	case -1:
		LOG_ERROR("%s(%d): Expected \'block\', \'randcrop\' or \'stretch\'", filename, ctx[1]);
		break;
	case 0:model->aug=AUG_BLOCK;break;
	case 1:model->aug=AUG_RANDCROP;break;
	case 2:model->aug=AUG_STRETCH;break;
	}
	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	ival=(int)round(fval);
	ASSERT_MSG(ival==2, "%s(%d): Only 2D input is supported, nDim = ", filename, ctx[1], ival);

	skip_ws(text, ctx);
	parse_number(filename, text, ctx, 10, &fval);
	model->input_shape[2]=(int)round(fval);
	
	skip_ws(text, ctx);
	if(text->data[*ctx]=='x')
	{
		++*ctx;
		skip_ws(text, ctx);
		parse_number(filename, text, ctx, 10, &fval);
		model->input_shape[3]=(int)round(fval);
	}
	else
		model->input_shape[3]=model->input_shape[2];

	ASSERT_MSG(model->input_shape[0]&&model->input_shape[2]&&model->input_shape[3], "Invalid input shape [%d, %d, %d, %d]", model->input_shape[0], model->input_shape[1], model->input_shape[2], model->input_shape[3]);

	ARRAY_ALLOC(Instruction, model->instructions, 0, 0, 0, 0);
	ARRAY_ALLOC(Buffer, model->buffers, 0, 0, 0, free_buffer);
	ARRAY_ALLOC(Variable, variables, 0, 0, 0, free_variable);

	for(;;)
	{
		skip_ws(text, ctx);
		match=match_kw(text, ctx, ksearch_global, COUNTOF(ksearch_global));
	
		switch(match)
		{
		case -1:
			LOG_ERROR("%s(%d): Expected \'cc\', \'conv\', \'save\', \'quantize\' or \'loss\'", filename, ctx[1]);
			break;
		case 0://cc/conv
		case 1:
			parse_conv(filename, text, ctx, model, variables, match);
			continue;
		case 2://save variable		not an instruction
			parse_save(filename, text, ctx, model, &variables);
			continue;
		case 3://quantize
			parse_quantize(filename, text, ctx, model, variables);
			continue;
		case 4://loss
			parse_loss(filename, text, ctx, model, variables);
			break;
		}
		break;
	}

	STR_COPY(model->src, text->data, *ctx);
	
	//print memory requirement
	size_t memusage=0;
	for(int k=0;k<(int)model->buffers->count;++k)
	{
		Buffer *buf=array_at(&model->buffers, k);
		size_t bufsize=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
		ASSERT_MSG(bufsize, "%s: Buffer %d size is zero, [%d %d %d %d]", filename, k, buf->shape[0], buf->shape[1], buf->shape[2], buf->shape[3]);
		if(using_gpu)
			bufsize*=sizeof(float);
		else
			bufsize*=sizeof(double);
		memusage+=bufsize;
	}
	acme_print_memsize(g_buf, G_BUF_SIZE, memusage);
	printf("Allocating %s of %s memory", g_buf, using_gpu?"GPU":"CPU");
	if(memusage>0x7FFFFFFF)
	{
		printf(", continue? [Y/N] ");
		char c=0;
		scanf(" %c", &c);
		if((c&0xDF)!='Y')
			exit(0);
	}
	else
		printf("\n");

	//allocate buffers
	size_t nParams=0, nGrad=0;
	int error=0;
	memusage=0;
	for(int k=0;k<(int)model->buffers->count;++k)
	{
		Buffer *buf=array_at(&model->buffers, k);
		size_t bufsize=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
		switch(buf->type)
		{
		case BUF_NORMAL:
		case BUF_NORMAL_SAVED:
			if(using_gpu)
			{
				buf->gpu_buf=p_clCreateBuffer(context, CL_MEM_READ_WRITE, bufsize*sizeof(float), 0, &error);
				CL_CHECK(error);
			}
			else
				buf->data=(double*)malloc(bufsize*sizeof(double));
			++memusage;
			break;
		case BUF_PARAM:
			buf->offset=nParams;
			nParams+=bufsize;
			break;
		case BUF_GRAD:
			buf->offset=nGrad;
			nGrad+=bufsize;
			break;
		default:
			LOG_ERROR("%s: Internal error", filename);
			break;
		}
	}
	ASSERT_MSG(nParams==nGrad, "%s: Different count of learnable params and gradient, nParams %d != nGrad %d", filename, nParams, nGrad);
	model->nparams=nParams;
	if(using_gpu)
	{
		model->gpu_params	=p_clCreateBuffer(context, CL_MEM_READ_WRITE, nParams*sizeof(float), 0, &error);	CL_CHECK(error);
		model->gpu_grad		=p_clCreateBuffer(context, CL_MEM_READ_WRITE, nParams*sizeof(float), 0, &error);	CL_CHECK(error);
		model->gpu_adam_m	=p_clCreateBuffer(context, CL_MEM_READ_WRITE, nParams*sizeof(float), 0, &error);	CL_CHECK(error);
		model->gpu_adam_v	=p_clCreateBuffer(context, CL_MEM_READ_WRITE, nParams*sizeof(float), 0, &error);	CL_CHECK(error);
	}
	else
	{
		ARRAY_ALLOC(double, model->cpu_params, 0, nParams, 0, 0);
		ARRAY_ALLOC(double, model->cpu_grad, 0, nGrad, 0, 0);
		ARRAY_ALLOC(double, model->cpu_adam_m, 0, nGrad, 0, 0);
		ARRAY_ALLOC(double, model->cpu_adam_v, 0, nGrad, 0, 0);
	}
	memusage+=4;

	skip_ws(text, ctx);
	if(!match_kw(text, ctx, (const char**)&kw_weights, 1))
	{
		for(size_t k=0;k<nParams;++k)
		{
			double *param=(double*)array_at(&model->cpu_params, k);
			skip_ws(text, ctx);
			parse_number(filename, text, ctx, 16, param);
		}
	}
	else
		init_model(model);

	skip_ws(text, ctx);
	ASSERT_MSG(*ctx==text->count, "%s(%d): Expected end of file", filename, ctx[1]);
	ASSERT_MSG(model->instructions&&model->instructions->count, "Model has no instructions");
	ASSERT_MSG(model->input_shape[1]==3, "Only 3 input channels are currently supported");
	
	array_free(&text);

	for(int kv=0;kv<(int)variables->count;++kv)
	{
		Variable *var=(Variable*)array_at(&variables, kv);
		Buffer *buf=(Buffer*)array_at(&model->buffers, var->buffer);
		ASSERT_MSG(buf->type==BUF_NORMAL||buf->type==BUF_NORMAL_SAVED, "Only normal data buffers can be saved (weights are saved anyway)");
		buf->type=BUF_NORMAL_SAVED;
	}
#if 0
	for(int ki=0;ki<(int)model->instructions->count;++ki)//X  the input is not a result
	{
		Instruction *inst=(Instruction*)array_at(&variables, ki);
		for(int kv=0;kv<(int)variables->count;++kv)
		{
			Variable *var=(Variable*)array_at(&variables, kv);
			if(var->buffer==inst->fwd_result)
			{
				inst->save_result=1;
				break;
			}
		}
	}
#endif
	array_free(&variables);

	{
		Instruction *inst=(Instruction*)array_at(&model->instructions, 0);
		model->input=inst->fwd_args[0];
	}
	return memusage;
}

int print_fhex(char *buf, size_t len, double x)
{
	int idx=0;

	if(x<0)
	{
		x=-x;
		buf[idx]='-';
		++idx;
	}
	if(!isfinite(x))
	{
		if(x!=x)
			memcpy(buf, "NaN", 3);
		else
			memcpy(buf, "inf", 3);
		idx+=3;
		return idx;
	}
	int start=idx;
	double fx=floor(x);
	x-=fx;
	unsigned long long llx=(unsigned long long)fx;
	while(llx)
	{
		buf[idx]=llx&15;
		buf[idx]+=buf[idx]<10?'0':'A';
		llx>>=4;
		++idx;
	}
	if(start<idx)//reverse the digits
	{
		for(char *p1=buf+start, *p2=buf+idx-1;p1<p2;++p1, --p2)
		{
			char t=*p1;
			*p1=*p2;
			*p2=t;
		}
		//memreverse(buf+start, start-idx, 1);
	}
	else
	{
		buf[idx]='0';
		++idx;
	}
	if(x)
	{
		buf[idx]='.';
		++idx;
		for(int k=0;k<12&&x;++k)
		{
			x*=16;
			int digit=(int)floor(x);
			x-=digit;
			buf[idx]=digit+(digit<10?'0':'A'-10);
			++idx;
		}
	}
	buf[idx]='\0';
	return idx;
}
void save_model(Model* model, int incremental, double loss)
{
	const char defaultname[]="model.txt";
	ArrayHandle filename, text;
	int printed;

	if(incremental)
		STR_COPY(filename, defaultname, sizeof(defaultname)-1);
	else
	{
		time_t t_now=time(0);
#ifdef _MSC_VER
		struct tm t_formatted={0}, *ts=&t_formatted;
		int error=localtime_s(ts, &t_now);
#else
		struct tm *ts=localtime(&t_now);
#endif
		printed=sprintf_s(g_buf, G_BUF_SIZE, "model-%04d%02d%02d-%02d%02d%02d-rmse%lf.txt", 1900+ts->tm_year, 1+ts->tm_mon, ts->tm_mday, ts->tm_hour, ts->tm_min, ts->tm_sec, loss);
		STR_COPY(filename, g_buf, printed);
	}
	STR_COPY(text, model->src->data, model->src->count);
	STR_APPEND(text, "\n\n", 2, 1);
	STR_APPEND(text, kw_weights, sizeof(kw_weights)-1, 1);
	STR_APPEND(text, "\n", 1, 1);

	size_t kp=0;
	float *gpu_params=0;
	if(using_gpu)
	{
		gpu_params=(float*)malloc(model->nparams*sizeof(float));
		int error=p_clEnqueueReadBuffer(commandqueue, model->gpu_params, CL_TRUE, 0, model->nparams*sizeof(float), gpu_params, 0, 0, 0);	CL_CHECK(error);
	}
	for(int kb=0;kb<(int)model->buffers->count;++kb)
	{
		Buffer *buf=(Buffer*)array_at(&model->buffers, kb);
		if(buf->type==BUF_PARAM)
		{
			for(int k0=0;k0<buf->shape[0];++k0)
			{
				for(int k1=0;k1<buf->shape[1];++k1)
				{
					for(int k2=0;k2<buf->shape[2];++k2)
					{
						if(using_gpu)
						{
							for(int k3=0;k3<buf->shape[3];++k3, ++kp)
							{
#ifdef FIXED_PREC
								double val=(double)gpu_params[kp]*(1./0x10000);
#else
								double val=gpu_params[kp];
#endif
								printed=print_fhex(g_buf, G_BUF_SIZE, val);
								STR_APPEND(text, "\t", 1, 1);
								STR_APPEND(text, g_buf, printed, 1);
							}
						}
						else
						{
							for(int k3=0;k3<buf->shape[3];++k3, ++kp)
							{
								double *val=(double*)array_at(&model->cpu_params, kp);
								printed=print_fhex(g_buf, G_BUF_SIZE, *val);
								STR_APPEND(text, "\t", 1, 1);
								STR_APPEND(text, g_buf, printed, 1);
							}
						}
						STR_APPEND(text, "\n", 1, 1);
					}
					STR_APPEND(text, "\n", 1, 1);
				}
				//STR_APPEND(text, "\n", 1, 1);
			}
			//STR_APPEND(text, "\n", 1, 1);
		}
	}
	//for(size_t k=0;k<model->params->count;++k)
	//{
	//	double *val=(double*)array_at(&model->params, k);
	//	printed=print_fhex(g_buf, G_BUF_SIZE, *val);
	//	STR_APPEND(text, "\n", 1, 1);
	//	STR_APPEND(text, g_buf, printed, 1);
	//}
	if(using_gpu)
		free(gpu_params);

	save_text(filename->data, text->data, text->count);
	array_free(&text);
	array_free(&filename);
}

void crosscorrelation2d(Model *model, Buffer *bufx, Buffer *buff, Buffer *bufb, Buffer *buf2)
{
#define FETCH(PTR, BUFFER)	PTR=BUFFER->type==BUF_NORMAL||BUFFER->type==BUF_NORMAL_SAVED?BUFFER->data:(BUFFER->type==BUF_PARAM?(double*)array_at(&model->cpu_params, BUFFER->offset):(double*)array_at(&model->cpu_grad, BUFFER->offset))
	double *data, *filt, *bias, *d2;
	FETCH(data, bufx);
	FETCH(filt, buff);
	FETCH(bias, bufb);
	FETCH(d2, buf2);
#undef	FETCH

	ASSERT_MSG(bufx->shape[0]==buf2->shape[0], "CC2D: Different batch size %d != %d", bufx->shape[0], buf2->shape[0]);
	ASSERT_MSG(bufx->shape[1]==buff->shape[1], "CC2D: Cin mismatch %d != %d", bufx->shape[1], buff->shape[1]);
	ASSERT_MSG(buf2->shape[1]==buff->shape[0], "CC2D: Cout mismatch, filter [%d %d %d %d], output [%d %d %d %d]",
		buff->shape[0], buff->shape[1], buff->shape[2], buff->shape[3],
		buf2->shape[0], buf2->shape[1], buf2->shape[2], buf2->shape[3]);
	ASSERT_MSG(bufx->shape[2]==buf2->shape[2]&&bufx->shape[3]==buf2->shape[3], "CC2D: Dimension mismatch %dx%d != %dx%d", bufx->shape[2], bufx->shape[3], buf2->shape[2], buf2->shape[3]);

	size_t ires=bufx->shape[2]*bufx->shape[3], isize=bufx->shape[1]*ires,
		osize=buf2->shape[1]*buf2->shape[2]*buf2->shape[3],
		K2=buff->shape[2]*buff->shape[3];
	int ypad=buff->shape[2]>>1, xpad=buff->shape[3]>>1;
	for(int kb=0;kb<buf2->shape[0];++kb)//for each sample in batch
	{
		double *outsample=d2+osize*kb;
		for(int ko=0;ko<buf2->shape[1];++ko)//for each output channel
		{
			for(int ky=0;ky<buf2->shape[2];++ky)//for output height
			{
				for(int kx=0;kx<buf2->shape[3];++kx, ++outsample)//for output width
				{
					double *inchannel=data+isize*kb;
					double sum=bias[ko];
					for(int ki=0;ki<buff->shape[1];++ki, inchannel+=ires)//for each input channel
					{
						double *kernel=filt+K2*(buff->shape[1]*ko+ki);
						for(int ky2=0;ky2<buff->shape[2];++ky2)//for each kernel row
						{
							unsigned yidx=ky+ky2-ypad;
							if((unsigned)yidx<(unsigned)buf2->shape[2])
							{
								for(int kx2=0;kx2<buff->shape[3];++kx2)//for each kernel value
								{
									unsigned xidx=kx+kx2-xpad;
									if((unsigned)xidx<(unsigned)buf2->shape[3])
										sum+=kernel[buff->shape[3]*ky2+kx2]*inchannel[bufx->shape[3]*yidx+xidx];
								}
							}
						}
					}
					*outsample=sum;
				}
			}
		}
	}
/*	for(int ko=0;ko<buff->shape[0];++ko)//for each filter
	{
		size_t k=ko*buf2->shape[2]*buf2->shape[3];
		for(int ky2=0;ky2<buf2->shape[2];++ky2)
		{
			for(int kx2=0;kx2<buf2->shape[3];++kx2, k)
				d2[k]=bias[ko];
		}
		k=ko*buf2->shape[2]*buf2->shape[3];
		for(int ki=0;ki<buff->shape[1];++ki)//for each channel
		{
			for(int ky=0;ky<buff->shape[2];++ky)
			{
				for(int kx=0;kx<buff->shape[3];++kx)
				{
					for(int ky2=0;ky2<buf2->shape[2];++ky2)
					{
						for(int kx2=0;kx2<buf2->shape[3];++kx2)
						{
						}
					}
				}
			}
		}
	}//*/
}
void dcrosscorrelation_dx_2d(Model *model, Buffer *bufx, Buffer *buff, Buffer *buf2)//filter Cin & Cout are swapped, so filt shape = [Ci,Co,Ky,Kx]
{
#define FETCH(PTR, BUFFER)	PTR=BUFFER->type==BUF_NORMAL||BUFFER->type==BUF_NORMAL_SAVED?BUFFER->data:(BUFFER->type==BUF_PARAM?(double*)array_at(&model->cpu_params, BUFFER->offset):(double*)array_at(&model->cpu_grad, BUFFER->offset))
	double *data, *filt, *d2;
	FETCH(data, bufx);
	FETCH(filt, buff);
	FETCH(d2, buf2);
#undef	FETCH

	//int temp=buff->shape[0];//X
	//buff->shape[0]=buff->shape[1];
	//buff->shape[1]=temp;

	ASSERT_MSG(bufx->shape[0]==buf2->shape[0], "Conv2D: Different batch size %d != %d", bufx->shape[0], buf2->shape[0]);
	ASSERT_MSG(bufx->shape[1]==buff->shape[0], "Conv2D: Cin mismatch %d != %d", bufx->shape[1], buff->shape[0]);
	ASSERT_MSG(buf2->shape[1]==buff->shape[1], "Conv2D: Cout mismatch %d != %d", buf2->shape[1], buff->shape[1]);
	ASSERT_MSG(bufx->shape[2]==buf2->shape[2]&&bufx->shape[3]==buf2->shape[3], "CC2D: Dimension mismatch %dx%d != %dx%d", bufx->shape[2], bufx->shape[3], buf2->shape[2], buf2->shape[3]);

	size_t ires=bufx->shape[2]*bufx->shape[3],
		isize=bufx->shape[1]*bufx->shape[2]*bufx->shape[3],
		osize=buf2->shape[1]*buf2->shape[2]*buf2->shape[3],
		K2=buff->shape[2]*buff->shape[3];
	int ypad=buff->shape[2]>>1, xpad=buff->shape[3]>>1;
	for(int kb=0;kb<buf2->shape[0];++kb)//for each sample in batch
	{
		double *outsample=d2+osize*kb;
		for(int ki=0;ki<buff->shape[1];++ki)//for each input channel
		{
			for(int ky=0;ky<buf2->shape[2];++ky)//for input height
			{
				for(int kx=0;kx<buf2->shape[3];++kx, ++outsample)//for input width
				{
					double *inchannel=data+isize*kb;
					double sum=0;
					for(int ko=0;ko<buff->shape[0];++ko, inchannel+=ires)//for each output channel
					{
						double *kernel=filt+K2*(buff->shape[1]*ko+ki);
						for(int kky=0;kky<buff->shape[2];++kky)//for kernel height
						{
							unsigned yidx=ky+kky-ypad;
							if((unsigned)yidx<(unsigned)buf2->shape[2])
							{
								for(int kkx=0;kkx<buff->shape[3];++kkx)//for kernel width
								{
									unsigned xidx=kx+kkx-xpad;
									if((unsigned)xidx<(unsigned)buf2->shape[3])
										sum+=kernel[buff->shape[3]*(buff->shape[2]-1-kky)+buff->shape[3]-1-kkx]*inchannel[bufx->shape[3]*yidx+xidx];
								}
							}
						}
					}
					*outsample=sum;
				}
			}
		}
	}
	//temp=buff->shape[0];//X
	//buff->shape[0]=buff->shape[1];
	//buff->shape[1]=temp;
}
//double dotsh2d(const double *fixed, const double *shifted, int Hf, int Wf, int Hs, int Ws, int dy, int dx)
//{
//	double sum=0;
//	for(int ky=0;ky<H;++ky)
//	{
//		int ky2=ky+dy;
//		if((unsigned)ky2<(unsigned)H)
//		{
//			for(int kx=0;kx<W;++kx)
//			{
//				int kx2=kx+dx;
//				if((unsigned)kx2<(unsigned)W)
//					sum+=fixed[W*ky+kx]*shifted[W*ky2+kx2];//integer multiplications can be optimized away
//			}
//		}
//	}
//	return sum;
//}
void dcrosscorrelation_dfilt_2d(Model *model, Buffer *buf_dL_dnet, Buffer *buf_x, Buffer *buf_dL_dfilt, int xpad, int ypad)
{
#define FETCH(PTR, BUFFER)	PTR=BUFFER->type==BUF_NORMAL||BUFFER->type==BUF_NORMAL_SAVED?BUFFER->data:(BUFFER->type==BUF_PARAM?(double*)array_at(&model->cpu_params, BUFFER->offset):(double*)array_at(&model->cpu_grad, BUFFER->offset))
	double *dL_dnet, *x, *dL_dfilt;
	FETCH(dL_dnet, buf_dL_dnet);
	FETCH(x, buf_x);
	FETCH(dL_dfilt, buf_dL_dfilt);
#undef	FETCH
	//int ypad=buf_dL_dfilt->shape[2]>>1, xpad=buf_dL_dfilt->shape[3]>>1;
	size_t
		netres=buf_dL_dnet->shape[2]*buf_dL_dnet->shape[3], bnetsize=buf_dL_dnet->shape[1]*netres,
		xres=buf_x->shape[2]*buf_x->shape[3], bxsize=buf_x->shape[1]*xres,
		ksize=buf_dL_dfilt->shape[2]*buf_dL_dfilt->shape[3];
	memset(dL_dfilt, 0, buf_dL_dfilt->shape[0]*buf_dL_dfilt->shape[1]*buf_dL_dfilt->shape[2]*buf_dL_dfilt->shape[3]*sizeof(double));//[Co,Ci,K,K]
	for(int kb=0;kb<buf_x->shape[0];++kb, dL_dnet+=bnetsize, x+=bxsize)//for each sample in batch
	{
		double *kernel=dL_dfilt;
		for(int ko=0;ko<buf_dL_dnet->shape[1];++ko)//for each channel Co in dL_dnet
		{
			for(int ki=0;ki<buf_x->shape[1];++ki)//for each channel Ci in x
			{
				for(int kky=0;kky<buf_dL_dfilt->shape[2];++kky)//for kernel height
				{
					for(int kkx=0;kkx<buf_dL_dfilt->shape[3];++kkx, ++kernel)//for kernel width
					{
						double sum=0;
						for(int ky=0;ky<buf_dL_dnet->shape[2];++ky)//for output height
						{
							int ky2=ky+kky-ypad;
							if((unsigned)ky2<(unsigned)buf_x->shape[2])
							{
								for(int kx=0;kx<buf_dL_dnet->shape[3];++kx)//for output width
								{
									int kx2=kx+kkx-xpad;
									if((unsigned)kx2<(unsigned)buf_x->shape[3])
										sum+=dL_dnet[netres*ko+buf_dL_dnet->shape[3]*ky+kx]*x[xres*ki+buf_x->shape[3]*ky2+kx2];
								}
							}
						}
						*kernel=sum;
					}
					//	*kernel+=dotsh2d(dL_dnet+netres*ko, x+xres*ki, buf_x->shape[2], buf_x->shape[3], kky-ypad, kkx-xpad);
				}
			}
		}
	}
}
void dcrosscorrelation_dbias_2d(Model *model, Buffer *buf_dL_dnet, Buffer *buf_dL_dbias)
{
#define FETCH(PTR, BUFFER)	PTR=BUFFER->type==BUF_NORMAL||BUFFER->type==BUF_NORMAL_SAVED?BUFFER->data:(BUFFER->type==BUF_PARAM?(double*)array_at(&model->cpu_params, BUFFER->offset):(double*)array_at(&model->cpu_grad, BUFFER->offset))
	double *dL_dnet, *dL_dbias;
	FETCH(dL_dnet, buf_dL_dnet);
	FETCH(dL_dbias, buf_dL_dbias);
#undef	FETCH
	//the sum is over [B, -, Ho, Wo]
	for(int kc=0;kc<buf_dL_dnet->shape[1];++kc)
	{
		dL_dbias[kc]=0;
		for(int kb=0;kb<buf_dL_dnet->shape[0];++kb)
		{
			double *ptr=dL_dnet+buf_dL_dnet->shape[3]*buf_dL_dnet->shape[2]*(buf_dL_dnet->shape[1]*kb+kc);
			for(int ky=0;ky<buf_dL_dnet->shape[2];++ky)
			{
				for(int kx=0;kx<buf_dL_dnet->shape[3];++kx, ++ptr)
					dL_dbias[kc]+=*ptr;
			}
		}
	}
}

const char *extensions[]=
{
	"jpg",
	"jpeg",
	"png",
};
unsigned char *load_nextimage(const char *path, ArrayHandle filenames, int *ki, int *iw, int *ih, int *warp)
{
	unsigned char *image=0;
	int nch=0, loaded=0;
	for(int ki2=*ki;;)
	{
		ArrayHandle *filename=(ArrayHandle*)array_at(&filenames, ki2);
		image=stbi_load(filename[0]->data, iw, ih, &nch, 4);
		if(image)
		{
			*ki=ki2;
			loaded=1;
			break;
		}
		*warp+=ki2+1>=(int)filenames->count;
		ki2=(ki2+1)%(int)filenames->count;
		if(ki2==*ki)
			break;
	}
	ASSERT_MSG(loaded, "No images found in \'%s\'", path);
	return image;
}
void assign_sample(const unsigned char *image, int iw, int ih, int px, int py, int kb, Buffer *input, float *gpubuf)//1:1
{
	double gain=1./255;
	for(int kc=0;kc<input->shape[1];++kc)//for each input channel
	{
		for(int ky=0;ky<input->shape[2];++ky)
		{
			if(gpubuf)
			{
				float *frow=gpubuf+input->shape[3]*(input->shape[2]*(input->shape[1]*kb+kc)+ky);
				int ky2=ky+py;
				if((unsigned)ky2<(unsigned)ih)
				{
					for(int kx=0;kx<input->shape[3];++kx)
					{
						int kx2=kx+px;
						if((unsigned)kx2<(unsigned)ih)
							frow[kx]=(float)(image[(iw*ky2+kx2)<<2|kc]*gain);
					}
				}
			}
			else
			{
				double *dstrow=input->data+input->shape[3]*(input->shape[2]*(input->shape[1]*kb+kc)+ky);
				int ky2=ky+py;
				if((unsigned)ky2<(unsigned)ih)
				{
					for(int kx=0;kx<input->shape[3];++kx)
					{
						int kx2=kx+px;
						if((unsigned)kx2<(unsigned)ih)
							dstrow[kx]=image[(iw*ky2+kx2)<<2|kc]*gain;
					}
				}
			}
		}
	}
}
int load_data(const char *path, ArrayHandle filenames, int *ki, int *kblock, Model *model)//returns number of times the whole dataset was read
{
	static unsigned char *image=0;
	static int ki0=-1, iw=0, ih=0;
	
	int warp=0;
	Buffer *input=(Buffer*)array_at(&model->buffers, model->input);
	size_t isize=input->shape[0]*input->shape[1]*input->shape[2]*input->shape[3];
	int error;
	if(using_gpu)
		memset(model->gpu_input, 0, isize*sizeof(float));
	else
		memset(input->data, 0, isize*sizeof(double));
	switch(model->aug)
	{
	case AUG_BLOCK://partition image to blocks
		{
			if(*ki!=ki0)
			{
				free(image);
				image=load_nextimage(path, filenames, ki, &iw, &ih, &warp);
				ki0=*ki;
			}
			int bcx=(iw+input->shape[3]-1)/input->shape[3],//ceil division
				bcy=(ih+input->shape[2]-1)/input->shape[2];
			int nblocks=bcx*bcy;
			int imsize=iw*ih;
			for(int kb=0;kb<input->shape[0];++kb, ++*kblock)//for each sample in batch
			{
				if(*kblock==nblocks)
				{
					warp+=*ki+1>=(int)filenames->count;
					*ki=(*ki+1)%(int)filenames->count;
					free(image);
					image=load_nextimage(path, filenames, ki, &iw, &ih, &warp);
					ki0=*ki, *kblock=0;
				}
				int px=(*kblock%bcx)*input->shape[3],
					py=(*kblock/bcx)*input->shape[2];
				assign_sample(image, iw, ih, px, py, kb, input, using_gpu?model->gpu_input:0);
			}
		}
		break;
	case AUG_RANDCROP://take one random block at 1:1 scale from each image
		for(int kb=0;kb<input->shape[0];++kb)//for each sample in batch
		{
			free(image);
			image=load_nextimage(path, filenames, ki, &iw, &ih, &warp);
			warp+=*ki+1>=(int)filenames->count;
			*ki=(*ki+1)%(int)filenames->count;
			ki0=*ki, *kblock=0;

			int px=iw<=input->shape[3]?0:rand()%(iw-input->shape[3]),
				py=ih<=input->shape[2]?0:rand()%(ih-input->shape[2]);
			assign_sample(image, iw, ih, px, py, kb, input, using_gpu?model->gpu_input:0);
		}
		break;
	case AUG_STRETCH://stretch image (smaller or larger dimensions), nearest for now
		{
			double gain=1./255;
			for(int kb=0;kb<input->shape[0];++kb)//for each sample in batch
			{
				free(image);
				image=load_nextimage(path, filenames, ki, &iw, &ih, &warp);
				warp+=*ki+1>=(int)filenames->count;
				*ki=(*ki+1)%(int)filenames->count;
				ki0=*ki, *kblock=0;

				for(int kc=0;kc<input->shape[1];++kc)//for each input channel
				{
					for(int ky=0;ky<input->shape[2];++ky)
					{
						if(using_gpu)
						{
							float *frow=model->gpu_input+input->shape[3]*(input->shape[2]*(input->shape[1]*kb+kc)+ky);
							int ky2=ky*(ih-1)/(input->shape[2]-1);
							for(int kx=0;kx<input->shape[3];++kx)
							{
								int kx2=kx*(iw-1)/(input->shape[3]-1);
								frow[kx]=(float)(image[(iw*ky2+kx2)<<2|kc]*gain);
							}
						}
						else
						{
							double *dstrow=input->data+input->shape[3]*(input->shape[2]*(input->shape[1]*kb+kc)+ky);
							int ky2=ky*(ih-1)/(input->shape[2]-1);
							for(int kx=0;kx<input->shape[3];++kx)
							{
								int kx2=kx*(iw-1)/(input->shape[3]-1);
								dstrow[kx]=image[(iw*ky2+kx2)<<2|kc]*gain;
							}
						}
					}
				}
			}
		}
		break;
	}
	if(using_gpu)
	{
#ifdef FIXED_PREC
		for(int k=0;k<isize;++k)
		{
			float *val=model->gpu_input+k;
			*(int*)val=*val*0x10000;
		}
#endif
		error=p_clEnqueueWriteBuffer(commandqueue, input->gpu_buf, CL_TRUE, 0, isize*sizeof(float), model->gpu_input, 0, 0, 0);
		CL_CHECK(error);
	}
	return warp;
}

typedef struct ConvInfoStruct//68 bytes
{	//	0	1	2	3	4	5	6	7		8		9			10			11	12	13		14		15		16
	int B,	Ci,	Hi,	Wi,	Co,	Kh,	Kw, xpad,	ypad,	xstride,	ystride,	Ho,	Wo, weight, bias,	epoch,	qlevels;//TODO support stride
} ConvInfo;
typedef struct AdamInfoStruct
{
	float lr, beta1, beta2, epsilon, gain1, gain2;//gain[i] = 1/(1-pow(beta[i], epoch))
} AdamInfo;
void fill_indices(Model *model, int ki, int ke, int qlevels, ConvInfo *cpu_indices, cl_mem gpu_indices)
{
	Instruction *inst=(Instruction*)array_at(&model->instructions, ki);
	Buffer
		*bufin	=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
		*buffilt=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
		*bufbias=(Buffer*)array_at(&model->buffers, inst->fwd_args[2]),
		*bufnet	=(Buffer*)array_at(&model->buffers, inst->fwd_result);
	cpu_indices->B		=bufin->shape[0];
	cpu_indices->Ci		=bufin->shape[1];
	cpu_indices->Hi		=bufin->shape[2];
	cpu_indices->Wi		=bufin->shape[3];
	cpu_indices->Co		=buffilt->shape[0];
	cpu_indices->Kh		=buffilt->shape[2];
	cpu_indices->Kw		=buffilt->shape[3];
	cpu_indices->xpad	=inst->info[0];
	cpu_indices->ypad	=inst->info[1];
	cpu_indices->xstride=inst->info[2];
	cpu_indices->ystride=inst->info[3];
	cpu_indices->Ho		=bufnet->shape[2];
	cpu_indices->Wo		=bufnet->shape[3];
	cpu_indices->weight	=(int)buffilt->offset;
	cpu_indices->bias	=(int)bufbias->offset;
	cpu_indices->epoch	=ke;
	cpu_indices->qlevels=qlevels;
	int error=p_clEnqueueWriteBuffer(commandqueue, gpu_indices, CL_TRUE, 0, sizeof(*cpu_indices), cpu_indices, 0, 0, 0);
	CL_CHECK(error);
}

typedef struct ResultCtxStruct
{
	int *rgb,
		nsaves,
		bcountx, bcounty,
		bsizex, bsizey,
		size;
} ResultCtx;
void result_save(ResultCtx *res, double loss, int idx)
{
	time_t rawtime;
	struct tm *info;
	time(&rawtime);
	info=localtime(&rawtime);
	if(idx)
		sprintf_s(g_buf, G_BUF_SIZE, "%04d%02d%02d-%02d%02d%02d-%d.PNG", 1900+info->tm_year, 1+info->tm_mon, info->tm_mday, info->tm_hour, info->tm_min, info->tm_sec, idx);
	else
		sprintf_s(g_buf, G_BUF_SIZE, "%04d%02d%02d-%02d%02d%02d-RMSE%lf.PNG", 1900+info->tm_year, 1+info->tm_mon, info->tm_mday, info->tm_hour, info->tm_min, info->tm_sec, loss);
	lodepng_encode_file(g_buf, res->rgb, res->nsaves*res->bcountx*res->bsizex, res->bcounty*res->bsizey, LCT_RGBA, 8);
}
int result_add(ResultCtx *res, Buffer *buf, int idx, int using_gpu, int force_save_loss)
{
	if(buf->type!=BUF_NORMAL_SAVED&&!force_save_loss)
		return 0;

	ASSERT_MSG(res->bcountx*res->bcounty>=buf->shape[0],
		"result_add: Batch size mismatch: expected B <= %d, got %d",
		res->bcountx*res->bcounty, buf->shape[0]);

	ASSERT_MSG(res->bsizex==buf->shape[3]&&res->bsizey==buf->shape[2]&&buf->shape[1]==3,
		"result_add: Buffer has wrong size: expected [3 %d %d], got [%d %d %d]",
		res->bsizey, res->bsizex, buf->shape[1], buf->shape[2], buf->shape[3]);

	int iw=res->nsaves*res->bcountx*res->bsizex;
	float *gpubuf=0;
	size_t size=0;
	if(using_gpu)
	{
		size=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
		gpubuf=(float*)malloc(size*sizeof(float));
		int error=p_clEnqueueReadBuffer(commandqueue, buf->gpu_buf, CL_TRUE, 0, size*sizeof(float), gpubuf, 0, 0, 0);	CL_CHECK(error);
	}
	for(int kb=0, ksrc=0;kb<buf->shape[0];++kb)//for each sample in batch
	{
		int kbx=kb%res->bcountx, kby=kb/res->bcountx;
		int *dst=res->rgb+iw*res->bsizey*kby+res->bsizex*(res->nsaves*kbx+idx);
		for(int kc=0;kc<buf->shape[1];++kc)//for each chanel
		{
			for(int ky=0;ky<buf->shape[2];++ky)//for src height
			{
				if(using_gpu)
				{
					for(int kx=0;kx<buf->shape[3];++kx, ++ksrc)//for src width
					{
						unsigned char *p=(unsigned char*)(dst+iw*ky+kx);
#ifdef FIXED_PREC
						int val=255*((int*)gpubuf)[ksrc]>>16;
						if(force_save_loss)
							val=(int)(127.5*0x10000)-val;
#else
						float val=255*gpubuf[ksrc];
						if(force_save_loss)
							val=127.5f-val;
#endif
						if(val<0)
							val=0;
						if(val>255)
							val=255;
						p[kc]=(unsigned char)val;
						//printf("%02X ", p[kc]);//
					}
				}
				else
				{
					for(int kx=0;kx<buf->shape[3];++kx, ++ksrc)//for src width
					{
						unsigned char *p=(unsigned char*)(dst+iw*ky+kx);
						double val=255*buf->data[ksrc];
						if(force_save_loss)
							val=127.5-val;
						if(val<0)
							val=0;
						if(val>255)
							val=255;
						p[kc]=(unsigned char)val;
					}
				}
			}
		}
	}
	if(using_gpu)
		free(gpubuf);
#if 0
	result_save(res, 0, 1+idx);//
#endif
	return 1;
}

double forward_gpu(Model *model, int is_test, int ke, int qlevels, ConvInfo *cpu_indices, cl_mem gpu_indices, ResultCtx *result)
{
	int error;
	cl_kernel kernel;
	size_t osize;
	double loss;
	int saveidx=0;
	Instruction *inst;

	if(is_test)
	{
		inst=(Instruction*)array_at(&model->instructions, 0);
		Buffer *input=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]);
		saveidx+=result_add(result, input, saveidx, 1, 0);
	}
	for(int ki=0;ki<(int)model->instructions->count;++ki)		//GPU forward
	{
		inst=(Instruction*)array_at(&model->instructions, ki);
		switch(inst->op)
		{
		case OP_CC2:
			{
				Buffer
					*bufin	=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*buffilt=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
					*bufbias=(Buffer*)array_at(&model->buffers, inst->fwd_args[2]),
					*bufnet	=(Buffer*)array_at(&model->buffers, inst->fwd_result);

				fill_indices(model, ki, ke, qlevels, cpu_indices, gpu_indices);

				kernel=kernels[OCL_cc2d];
				osize=bufnet->shape[0]*bufnet->shape[1]*bufnet->shape[2]*bufnet->shape[3];
				error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufin->gpu_buf);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &model->gpu_params);	CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufnet->gpu_buf);	CL_CHECK(error);
				error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);
				p_clFlush(commandqueue);
				p_clFinish(commandqueue);

				if(is_test)
					saveidx+=result_add(result, bufnet, saveidx, 1, 0);

				//preview_cl_buffer(bufin, 0, 0, 0);
				//preview_cl_buffer(buffilt, model->gpu_params, cpu_indices.weight, 0);
				//preview_cl_buffer(buffilt, model->gpu_params, cpu_indices.bias, 0);
				//preview_cl_buffer(bufnet, 0, 0, 0);
#if 0
				{
					float *buffer=(float*)malloc(osize*sizeof(float));
					error=p_clEnqueueReadBuffer(commandqueue, bufnet->gpu_buf, CL_TRUE, 0, osize, buffer, 0, 0, 0);	CL_CHECK(error);
					if(osize>200)
						osize=200;
					for(size_t k=0;k<osize;++k)
					{
						printf("%g ", (double)buffer[k]);
						if(!isfinite(buffer[k]))
						{
							int k2, kb, kc, ky, kx;
							k2=k;
							kx=k2%bufnet->shape[3], k2/=bufnet->shape[3];
							ky=k2%bufnet->shape[2], k2/=bufnet->shape[2];
							kc=k2%bufnet->shape[1], k2/=bufnet->shape[1];
							kb=k2;
							printf("%d [%d %d %d %d] %016llX = %f\n", k, kb, kc, ky, kx, buffer[k], buffer[k]);
							LOG_ERROR("");
						}
					}
					printf("\n");
				}
#endif
			}
			break;
		case OP_CONV2:
			LOG_ERROR("Conv2d is not supported yet. Use CC2D.");
			break;
		case OP_LRELU:
		case OP_RELU:
		case OP_QUANTIZER:
			{
				Buffer
					*bufin	=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*bufout	=(Buffer*)array_at(&model->buffers, inst->fwd_result);
				osize=bufout->shape[0]*bufout->shape[1]*bufout->shape[2]*bufout->shape[3];
				kernel=0;
				switch(inst->op)
				{
				case OP_LRELU:		kernel=kernels[OCL_lrelu	];break;
				case OP_RELU:		kernel=kernels[OCL_relu		];break;
				case OP_QUANTIZER:	kernel=kernels[is_test?OCL_quantizer_test:OCL_quantizer_train];break;
				}
				error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufin->gpu_buf);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufout->gpu_buf);	CL_CHECK(error);
				error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);
				p_clFlush(commandqueue);
				p_clFinish(commandqueue);

				if(is_test)
					saveidx+=result_add(result, bufout, saveidx, 1, 0);
			}
			break;
		case OP_MSE:
			{
				Buffer
					*buf1	=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*buf2	=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
					*bufdiff=(Buffer*)array_at(&model->buffers, inst->fwd_result);

				kernel=kernels[OCL_loss_MSE];
				osize=bufdiff->shape[0]*bufdiff->shape[1]*bufdiff->shape[2]*bufdiff->shape[3];
				error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf1->gpu_buf);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf2->gpu_buf);		CL_CHECK(error);
				error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufdiff->gpu_buf);	CL_CHECK(error);
				error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);
				p_clFlush(commandqueue);
				p_clFinish(commandqueue);

				if(is_test)
					saveidx+=result_add(result, bufdiff, saveidx, 1, 1);

				error=p_clEnqueueReadBuffer(commandqueue, bufdiff->gpu_buf, CL_TRUE, 0, osize*sizeof(float), model->gpu_output, 0, 0, 0);	CL_CHECK(error);
				double maxloss=0;
				size_t maxidx=0;
				loss=0;
				for(size_t k=0;k<osize;++k)
				{
#ifdef FIXED_PREC
					double lk=((int*)model->gpu_output)[k]*(1./0x10000);
#else
					double lk=model->gpu_output[k];
#endif
					double l2=(double)lk*lk;
					//if(!isfinite(l2))
					//	LOG_ERROR("Infinite loss at epoch %d, at %lld, size %llf\n0x%016llX = %g\n", ke, k, osize, lk, (double)model->gpu_output[k]);
					if(maxloss<l2)
						maxloss=l2, maxidx=k;
					loss+=l2;
				}
				ASSERT_MSG(isfinite(loss), "\nInfinite loss at epoch %d, idx %lld, size=%lld\n0x%016llX = %g\n", ke, maxidx, osize, maxloss, maxloss);
				loss/=osize<<1;
			}
			break;
		case OP_MS_SSIM:
			LOG_ERROR("MS-SSIM is not supported yet. Use MSE.");
			break;
		default:
			LOG_ERROR("Unrecognized instruction type %d.", inst->op);
			break;
		}
	}
	return loss;
}
double forward_cpu(Model *model, int is_test, ResultCtx *result)
{
	double loss;
	int saveidx=0;
	Instruction *inst;

	//CPU forward
	if(is_test)
	{
		inst=(Instruction*)array_at(&model->instructions, 0);
		Buffer *input=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]);
		saveidx+=result_add(result, input, saveidx, 1, 0);
	}
	for(int ki=0;ki<(int)model->instructions->count;++ki)
	{
		inst=(Instruction*)array_at(&model->instructions, ki);
#ifdef DEBUG_AUTOGRAD
		const char *a=0;
		switch(inst->op)
		{
		case OP_CC2			:a="OP_CC2      ";break;
		case OP_CONV2		:a="OP_CONV2    ";break;
		case OP_RELU		:a="OP_RELU     ";break;
		case OP_LRELU		:a="OP_LRELU    ";break;
		case OP_QUANTIZER	:a="OP_QUANTIZER";break;
		case OP_MSE			:a="OP_MSE      ";break;
		case OP_MS_SSIM		:a="OP_MS_SSIM  ";break;
		}
		printf("fwd %d/%d %s\n", ki+1, (int)model->instructions->count, a);//
#endif
		switch(inst->op)
		{
		case OP_CC2:
			{
				Buffer
					*bufx=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*buff=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
					*bufb=(Buffer*)array_at(&model->buffers, inst->fwd_args[2]),
					*bufn=(Buffer*)array_at(&model->buffers, inst->fwd_result);
//#ifdef DEBUG_AUTOGRAD
//					printf("fwd %d %d %d result %d, out shape [%d %d %d %d]\n",
//						inst->fwd_args[0], inst->fwd_args[1], inst->fwd_args[2], inst->fwd_result,
//						bufn->shape[0], bufn->shape[1], bufn->shape[2], bufn->shape[3]);//
//#endif
				crosscorrelation2d(model, bufx, buff, bufb, bufn);

				if(is_test)
					saveidx+=result_add(result, bufn, saveidx, 0, 0);
#if 0
				{
					for(size_t k=0, osize=bufn->shape[0]*bufn->shape[1]*bufn->shape[2]*bufn->shape[3];k<osize;++k)
					{
						float val=bufn->data[k];
						if(!isfinite(val)||val!=val)
						{
							int k0, kb, kc, ky, kx;
							k0=(int)k;
							kx=k%bufn->shape[3], k/=bufn->shape[3];
							ky=k%bufn->shape[2], k/=bufn->shape[2];
							kc=k%bufn->shape[1], k/=bufn->shape[1];
							kb=(int)k;
							printf("%d [%d %d %d %d] %f\n", k0, kb, kc, ky, kx, bufn->data[k]);
							break;
						}
					}
				}
#endif
			}
			continue;
		case OP_CONV2:
			LOG_ERROR("Conv2d is not supported yet. Use CC2D.");
			//{
			//	Buffer
			//		*bufx=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
			//		*buff=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
			//		*bufb=(Buffer*)array_at(&model->buffers, inst->fwd_args[2]),
			//		*buf2=(Buffer*)array_at(&model->buffers, inst->fwd_result);
			//	conv2d(model, bufx, buff, bufb, buf2, 1);
			//}
			continue;
		case OP_LRELU:
			{
				Buffer
					*src=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*dst=(Buffer*)array_at(&model->buffers, inst->fwd_result);
				ASSERT_MSG(src->type==BUF_NORMAL||src->type==BUF_NORMAL_SAVED, "LReLU cannot be applied to learnable parameters, src->type = %d", src->type);
				ASSERT_MSG(dst->type==BUF_NORMAL||dst->type==BUF_NORMAL_SAVED, "LReLU cannot be applied to learnable parameters, src->type = %d", dst->type);
				LeakyReLU(src->data, dst->data, src->shape[0]*src->shape[1]*src->shape[2]*src->shape[3]);
			}
			continue;
		case OP_RELU:
			{
				Buffer
					*src=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*dst=(Buffer*)array_at(&model->buffers, inst->fwd_result);
				ASSERT_MSG(src->type==BUF_NORMAL||src->type==BUF_NORMAL_SAVED, "ReLU cannot be applied to learnable parameters, src->type = %d", src->type);
				ASSERT_MSG(dst->type==BUF_NORMAL||dst->type==BUF_NORMAL_SAVED, "ReLU cannot be applied to learnable parameters, src->type = %d", dst->type);
				ReLU(src->data, dst->data, src->shape[0]*src->shape[1]*src->shape[2]*src->shape[3]);
			}
			continue;
		case OP_QUANTIZER:
			{
				Buffer
					*src=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*dst=(Buffer*)array_at(&model->buffers, inst->fwd_result);
				ASSERT_MSG(src->type==BUF_NORMAL||src->type==BUF_NORMAL_SAVED, "Quantizer cannot be applied to learnable parameters, src->type = %d", src->type);
				ASSERT_MSG(dst->type==BUF_NORMAL||dst->type==BUF_NORMAL_SAVED, "Quantizer cannot be applied to learnable parameters, src->type = %d", dst->type);
				int size=src->shape[0]*src->shape[1]*src->shape[2]*src->shape[3];
				if(is_test)
				{
					double gain=1./(inst->info[0]-1);
					for(size_t k=0;k<size;++k)
					{
						double x=(inst->info[0]-1)*src->data[k];
						x=round(x);
						if(x<0)
							x=0;
						if(x>inst->info[0]-1)
							x=inst->info[0]-1;
						x*=gain;
						dst->data[k]=x;
					}
					saveidx+=result_add(result, dst, saveidx, 0, 0);
				}
				else
				{
					double gain=1./((double)(inst->info[0]-1)*RAND_MAX);
					for(size_t k=0;k<size;++k)
						dst->data[k]=src->data[k]+gain*(rand()-(RAND_MAX>>1));
				}
			}
			continue;
		case OP_MSE:
			{
				Buffer
					*s1=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
					*s2=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
					*dst=(Buffer*)array_at(&model->buffers, inst->fwd_result);
				ASSERT_MSG(s1->type==BUF_NORMAL||s1->type==BUF_NORMAL_SAVED, "MSE cannot be applied to learnable parameters, src->type = %d", s1->type);
				ASSERT_MSG(s2->type==BUF_NORMAL||s2->type==BUF_NORMAL_SAVED, "MSE cannot be applied to learnable parameters, src->type = %d", s2->type);
				ASSERT_MSG(dst->type==BUF_NORMAL||dst->type==BUF_NORMAL_SAVED, "MSE cannot be applied to learnable parameters, src->type = %d", dst->type);
				size_t count=s1->shape[0]*s1->shape[1]*s1->shape[2]*s1->shape[3];
				loss=0;
				for(size_t k=0;k<count;++k)
				{
					dst->data[k]=s1->data[k]-s2->data[k];
					loss+=dst->data[k]*dst->data[k];
				}
				loss/=count<<1;

				if(is_test)
					saveidx+=result_add(result, dst, saveidx, 0, 1);
			}
			break;
		case OP_MS_SSIM:
			LOG_ERROR("MS-SSIM is not supported yet. Use MSE.");
			break;
		default:
			LOG_ERROR("Unrecognized instruction type %d.", inst->op);
			break;
		}
		break;
	}
	return loss;
}

Model model={0};
ConvInfo cpu_indices={0};
AdamInfo adam_params={0};
#endif

typedef enum CMDArgsEnum
{
	ARG_PROGRAM,

	ARG_DEVICE,
	ARG_EPOCHS,
	ARG_LR,
	ARG_BATCHSIZE,
	ARG_MODEL,

	ARG_COUNT,
} CMDArgs;
int main(int argc, char **argv)
{
	set_console_buffer_size(120, 2000);
	//system("cd");//
	
	//automatic differentiation in 2D
#if 1
	printf(
		"ACME-ML\t\tbuild %s %s\n", __DATE__, __TIME__);
	if(argc!=ARG_COUNT-1&&argc!=ARG_COUNT)
	{
		printf(
			"Usage:\n"
			"  program  device  epochs  LR  batchSize  [modelName.txt]\n"
			"\n"
			"device: GPU or CPU\n"
			"If model name is not given, will attempt to load the incremental \'model.txt\'\n"
			"\n"
		);
#ifdef _MSC_VER
		pause();
#endif
		return 1;
	}
	int error;
	if(!strcmp(argv[ARG_DEVICE], "GPU"))
	{
		using_gpu=1;
		ocl_init("cl_kernels.h");
	}
	else if(strcmp(argv[ARG_DEVICE], "CPU"))//not CPU
	{
		printf("Invalid device name \'%s\', expected GPU or CPU", argv[ARG_DEVICE]);
#ifdef _MSC_VER
		pause();
#endif
		return 1;
	}

	//cc test
#if 0
	{
		ConvInfo indices=
		{
			1,
			1, 4, 4,//Ci,Hi,Wi
			1, 3, 3,//Co,Kh,Kw
			1, 1, 1, 1,//pad, stride
			4, 4,//Ho,Wo
			0, 9,//params
			0, 10,
		};
		size_t osize=indices.Wo*indices.Ho;
		float input[]=
		{
			1, 1, 0, 0,
			1, 1, 0, 0,
			0, 0, 1, 1,
			0, 0, 1, 1,
		};
		float params[]=
		{
			1.f/9, 1.f/9, 1.f/9,
			1.f/9, 1.f/9, 1.f/9,
			1.f/9, 1.f/9, 1.f/9,
			0,
		};
		float output[16];
		cl_mem gpu_indices, gpu_input, gpu_params, gpu_output;
		gpu_indices	=p_clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(indices), 0, &error);	CL_CHECK(error);
		gpu_input	=p_clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(input), 0, &error);		CL_CHECK(error);
		gpu_params	=p_clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(params), 0, &error);	CL_CHECK(error);
		gpu_output	=p_clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(output), 0, &error);	CL_CHECK(error);
		error=p_clEnqueueWriteBuffer(commandqueue, gpu_indices, CL_TRUE, 0, sizeof(indices), &indices, 0, 0, 0);CL_CHECK(error);
		error=p_clEnqueueWriteBuffer(commandqueue, gpu_input	, CL_TRUE, 0, sizeof(input), input, 0, 0, 0);	CL_CHECK(error);
		error=p_clEnqueueWriteBuffer(commandqueue, gpu_params	, CL_TRUE, 0, sizeof(params), params, 0, 0, 0);	CL_CHECK(error);
		cl_kernel kernel=kernels[OCL_cc2d];
		error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);CL_CHECK(error);
		error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_input);	CL_CHECK(error);
		error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_params);	CL_CHECK(error);
		error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &gpu_output);	CL_CHECK(error);
		error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);
		error=p_clFlush(commandqueue);	CL_CHECK(error);
		error=p_clFinish(commandqueue);	CL_CHECK(error);
		error=p_clEnqueueReadBuffer(commandqueue, gpu_output, CL_TRUE, 0, sizeof(output), output, 0, 0, 0);	CL_CHECK(error);
		print_2d(output, indices.Ho, indices.Wo, 0, "Output:");
		pause();
		exit(0);
	}
#endif
	int epoch=atoi(argv[ARG_EPOCHS]);
	double lr=atof(argv[ARG_LR]);
	model.input_shape[0]=atoi(argv[ARG_BATCHSIZE]);//Batch size
	int nallocs=(int)parse_model(argc==ARG_MODEL?"model.txt":argv[ARG_MODEL], &model);

	//for(int k=0;k<model.buffers->count;++k)
	//{
	//	Buffer *buf=(Buffer*)array_at(&model.buffers, k);
	//	printf("\t%p\n", buf->offset);
	//}
	int nlayers=0;
	for(int ki=0;ki<(int)model.instructions->count;++ki)
	{
		Instruction *inst=(Instruction*)array_at(&model.instructions, ki);
		nlayers+=inst->op==OP_CC2||inst->op==OP_CONV2;
	}

	printf("%d buffers\n%lld parameters, %d layers\n%d epochs, lr=%g, batchSize=%d\n", nallocs, model.nparams, nlayers, epoch, lr, model.input_shape[0]);

	//read dataset directory
	ArrayHandle filenames=get_filenames(model.trainpath->data, extensions, COUNTOF(extensions));
	ASSERT_MSG(filenames->count, "No images found in train path \'%s\'", (char*)model.trainpath->data);

	//train
	double av_loss=0, loss=0,
		beta1=0.94, beta2=0.9878, epsilon=1e-8,//adam optimizer
		beta1_t=beta1, beta2_t=beta2;
	int nbatches=0;
	int ki=0,//points at current image to load
		kblock=0;//points at current image block in case of block augmentation
	double timestamp1=time_ms(), timestamp2;
	int ke=0;
	cl_mem gpu_indices=0, gpu_adam_params=0;
	int qlevels=0;
	size_t osize;
	cl_kernel kernel;
	if(using_gpu)
	{
		Buffer *buf;
		size_t size;
		Instruction *inst;

		buf=(Buffer*)array_at(&model.buffers, model.input);
		size=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
		model.gpu_input=(float*)malloc(size*sizeof(float));

		inst=(Instruction*)array_at(&model.instructions, model.instructions->count-1);
		buf=(Buffer*)array_at(&model.buffers, inst->fwd_result);
		size=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
		model.gpu_output=(float*)malloc(size*sizeof(float));

		gpu_indices		=p_clCreateBuffer(context, CL_MEM_READ_ONLY, 17*sizeof(int), 0, &error);	CL_CHECK(error);
		gpu_adam_params	=p_clCreateBuffer(context, CL_MEM_READ_ONLY, 6*sizeof(float), 0, &error);	CL_CHECK(error);

		for(int ki=0;ki<(int)model.instructions->count;++ki)
		{
			Instruction *inst=(Instruction*)array_at(&model.instructions, ki);
			if(inst->op==OP_QUANTIZER)
			{
				qlevels=inst->info[0];
				break;
			}
		}
	}
	for(;;)
	{
		//load data
		int epoch_inc=load_data((char*)model.trainpath, filenames, &ki, &kblock, &model);
		if(epoch_inc)
		{
			timestamp2=time_ms();
			++ke;
			av_loss/=nbatches;
			double psnr=20*log10(255/av_loss);
			acme_strftime(g_buf, G_BUF_SIZE, (timestamp2-timestamp1)/1000);
			printf("\t\t\t\t\rEpoch %3d RMSE %16.12lf PSNR %13.9lf %10lf minutes %s\n", ke, av_loss, psnr, (timestamp2-timestamp1)/60000, g_buf);
			
			save_model(&model, 1, av_loss);

			if(ke>=epoch)
				break;
			
			av_loss=0, nbatches=0;

			beta1_t*=beta1;//Adam Optimizer
			beta2_t*=beta2;
		}

		if(using_gpu)
		{
			loss=forward_gpu(&model, 0, ke, qlevels, &cpu_indices, gpu_indices, 0);

			for(int ki=(int)model.instructions->count-1;ki>=0;--ki)		//GPU backward
			{
				Instruction *inst=(Instruction*)array_at(&model.instructions, ki);
				switch(inst->op)
				{
				case OP_CC2:
					{
						Buffer
							*dL_dnet	=(Buffer*)array_at(&model.buffers, inst->bwd_args[0]),
							*filt		=(Buffer*)array_at(&model.buffers, inst->bwd_args[1]),
							*in			=(Buffer*)array_at(&model.buffers, inst->bwd_args[2]),
							*dL_din		=(Buffer*)array_at(&model.buffers, inst->bwd_results[0]),
							*dL_dfilt	=(Buffer*)array_at(&model.buffers, inst->bwd_results[1]),
							*dL_dbias	=(Buffer*)array_at(&model.buffers, inst->bwd_results[2]);

						kernel=kernels[OCL_cc2d_grad_in];
						osize=dL_din->shape[0]*dL_din->shape[1]*dL_din->shape[2]*dL_din->shape[3];
						error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &dL_dnet->gpu_buf);	CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &model.gpu_params);	CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &dL_din->gpu_buf);	CL_CHECK(error);
						error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);

						kernel=kernels[OCL_cc2d_grad_filt];
						osize=dL_dfilt->shape[0]*dL_dfilt->shape[1]*dL_dfilt->shape[2]*dL_dfilt->shape[3];
						error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &dL_dnet->gpu_buf);	CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &in->gpu_buf);		CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &model.gpu_grad);		CL_CHECK(error);
						error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);

						kernel=kernels[OCL_cc2d_grad_bias];
						osize=dL_dbias->shape[0]*dL_dbias->shape[1]*dL_dbias->shape[2]*dL_dbias->shape[3];
						error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &dL_dnet->gpu_buf);	CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &model.gpu_grad);		CL_CHECK(error);
						error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);

						p_clFlush(commandqueue);
						p_clFinish(commandqueue);
					}
					break;
				case OP_CONV2:
					LOG_ERROR("Conv2d is not supported yet. Use CC2D.");
					break;
				case OP_LRELU:
				case OP_RELU:
					{
						Buffer
							*bufdL_dout	=(Buffer*)array_at(&model.buffers, inst->bwd_args[0]),
							*bufout		=(Buffer*)array_at(&model.buffers, inst->bwd_args[1]),
							*bufdL_din	=(Buffer*)array_at(&model.buffers, inst->bwd_results[0]);

						kernel=0;
						switch(inst->op)
						{
						case OP_LRELU:	kernel=kernels[OCL_lrelu_grad	];break;
						case OP_RELU:	kernel=kernels[OCL_relu_grad	];break;
						}
						for(int ki2=ki-1;ki2>=0;--ki2)//look for a conv instruction before, and take its indices
						{
							inst=(Instruction*)array_at(&model.instructions, ki2);
							if(inst->op==OP_CC2||inst->op==OP_CONV2)
							{
								fill_indices(&model, ki2, ke, qlevels, &cpu_indices, gpu_indices);
								break;
							}
						}
						osize=bufdL_din->shape[0]*bufdL_din->shape[1]*bufdL_din->shape[2]*bufdL_din->shape[3];
						error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_indices);		CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufdL_dout->gpu_buf);CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufout->gpu_buf);	CL_CHECK(error);
						error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufdL_din->gpu_buf);	CL_CHECK(error);
						error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &osize, 0, 0, 0, 0);	CL_CHECK(error);
						p_clFlush(commandqueue);
						p_clFinish(commandqueue);
					}
					break;
				case OP_QUANTIZER://identity
				case OP_MSE://identity
					break;
				case OP_MS_SSIM:
					LOG_ERROR("MS-SSIM is not supported yet. Use MSE.");
					break;
				default:
					LOG_ERROR("Unrecognized instruction type %d.", inst->op);
					break;
				}
			}
			adam_params.lr=(float)lr;
			adam_params.beta1=(float)beta1;
			adam_params.beta2=(float)beta2;
			adam_params.epsilon=(float)epsilon;
			adam_params.gain1=(float)(1/(1-beta1_t));
			adam_params.gain2=(float)(1/(1-beta2_t));
#ifdef FIXED_PREC
			for(int k=0;k<sizeof(adam_params)/sizeof(float);++k)
			{
				float *val=(float*)&adam_params+k;
				*(int*)val=*val*0x10000;
			}
#endif
			error=p_clEnqueueWriteBuffer(commandqueue, gpu_adam_params, CL_TRUE, 0, sizeof(adam_params), &adam_params, 0, 0, 0);	CL_CHECK(error);
			kernel=kernels[OCL_opt_adam];
			error=p_clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_adam_params);	CL_CHECK(error);
			error=p_clSetKernelArg(kernel, 1, sizeof(cl_mem), &model.gpu_grad);		CL_CHECK(error);
			error=p_clSetKernelArg(kernel, 2, sizeof(cl_mem), &model.gpu_adam_m);	CL_CHECK(error);
			error=p_clSetKernelArg(kernel, 3, sizeof(cl_mem), &model.gpu_adam_v);	CL_CHECK(error);
			error=p_clSetKernelArg(kernel, 4, sizeof(cl_mem), &model.gpu_params);	CL_CHECK(error);
			error=p_clEnqueueNDRangeKernel(commandqueue, kernel, 1, 0, &model.nparams, 0, 0, 0, 0);	CL_CHECK(error);
			p_clFlush(commandqueue);
			p_clFinish(commandqueue);
		}
		else
		{
			loss=forward_cpu(&model, 0, 0);

			//CPU backward
			for(int ki=(int)model.instructions->count-1;ki>=0;--ki)
			{
				Instruction *inst=(Instruction*)array_at(&model.instructions, ki);
#ifdef DEBUG_AUTOGRAD
				const char *a=0;
				switch(inst->op)
				{
				case OP_CC2			:a="OP_CC2      ";break;
				case OP_CONV2		:a="OP_CONV2    ";break;
				case OP_RELU		:a="OP_RELU     ";break;
				case OP_LRELU		:a="OP_LRELU    ";break;
				case OP_QUANTIZER	:a="OP_QUANTIZER";break;
				case OP_MSE			:a="OP_MSE      ";break;
				case OP_MS_SSIM		:a="OP_MS_SSIM  ";break;
				}
				printf("bwd %d/%d %s\n", ki+1, (int)model.instructions->count, a);//
#endif
				switch(inst->op)
				{
				case OP_CC2:
					{
						Buffer
							*dL_dnet	=(Buffer*)array_at(&model.buffers, inst->bwd_args[0]),
							*filt		=(Buffer*)array_at(&model.buffers, inst->bwd_args[1]),
							*x			=(Buffer*)array_at(&model.buffers, inst->bwd_args[2]),
							*dL_dx		=(Buffer*)array_at(&model.buffers, inst->bwd_results[0]),
							*dL_dfilt	=(Buffer*)array_at(&model.buffers, inst->bwd_results[1]),
							*dL_dbias	=(Buffer*)array_at(&model.buffers, inst->bwd_results[2]);
						dcrosscorrelation_dx_2d(&model, dL_dnet, filt, dL_dx);
						dcrosscorrelation_dfilt_2d(&model, dL_dnet, x, dL_dfilt, inst->info[0], inst->info[1]);
						dcrosscorrelation_dbias_2d(&model, dL_dnet, dL_dbias);
					}
					break;
				case OP_CONV2:
					LOG_ERROR("Conv2d is not supported yet. Use CC2D.");
					break;
				case OP_LRELU:
					{
						Buffer
							*dL_dout=(Buffer*)array_at(&model.buffers, inst->bwd_args[0]),
							*out	=(Buffer*)array_at(&model.buffers, inst->bwd_args[1]),
							*dL_din	=(Buffer*)array_at(&model.buffers, inst->bwd_results[0]);
						ASSERT_MSG(dL_dout->type==BUF_NORMAL||dL_dout->type==BUF_NORMAL_SAVED, "LRELU cannot be applied to learnable parameters, src->type = %d", dL_dout->type);
						ASSERT_MSG(out->type==BUF_NORMAL||out->type==BUF_NORMAL_SAVED, "LRELU cannot be applied to learnable parameters, src->type = %d", out->type);
						ASSERT_MSG(dL_din->type==BUF_NORMAL||dL_din->type==BUF_NORMAL_SAVED, "LRELU cannot be applied to learnable parameters, src->type = %d", dL_din->type);
						size_t size=out->shape[0]*out->shape[1]*out->shape[2]*out->shape[3];
						for(int k=0;k<size;++k)//dL_dout .* act'(out)
						{
							dL_din->data[k]=dL_dout->data[k];
							if(out->data[k]<0)
								dL_din->data[k]*=0.01;
						}
					}
					break;
				case OP_RELU:
					{
						Buffer
							*dL_dx2	=(Buffer*)array_at(&model.buffers, inst->bwd_args[0]),
							*x2		=(Buffer*)array_at(&model.buffers, inst->bwd_args[1]),
							*dL_dnet=(Buffer*)array_at(&model.buffers, inst->bwd_results[0]);
						ASSERT_MSG(dL_dx2->type==BUF_NORMAL||dL_dx2->type==BUF_NORMAL_SAVED, "LRELU cannot be applied to learnable parameters, src->type = %d", dL_dx2->type);
						ASSERT_MSG(x2->type==BUF_NORMAL||x2->type==BUF_NORMAL_SAVED, "LRELU cannot be applied to learnable parameters, src->type = %d", x2->type);
						ASSERT_MSG(dL_dnet->type==BUF_NORMAL||dL_dnet->type==BUF_NORMAL_SAVED, "LRELU cannot be applied to learnable parameters, src->type = %d", dL_dnet->type);
						size_t size=x2->shape[0]*x2->shape[1]*x2->shape[2]*x2->shape[3];
						for(int k=0;k<size;++k)//dL_dx2 .* act'(x2)
						{
							double x=x2->data[k];
							dL_dnet->data[k]=x<0?0:dL_dx2->data[k];
						}
					}
					break;
				case OP_QUANTIZER://identity
				case OP_MSE://identity
					break;
				case OP_MS_SSIM:
					LOG_ERROR("MS-SSIM is not supported yet.");
					break;
				default:
					LOG_ERROR("Unrecognized instruction %d.", inst->op);
					break;
				}
			}

			mix_inplace((double*)model.cpu_adam_m->data, (double*)model.cpu_grad->data, beta1, model.cpu_grad->count);
			sq_inplace((double*)model.cpu_grad->data, model.cpu_grad->count);
			mix_inplace((double*)model.cpu_adam_v->data, (double*)model.cpu_grad->data, beta2, model.cpu_grad->count);

			double gain1=1/(1-beta1_t), gain2=1/(1-beta2_t);
			for(int k=0;k<model.cpu_grad->count;++k)//https://optimization.cbe.cornell.edu/index.php?title=Adam
			{
				double
					mhat=((double*)model.cpu_adam_m->data)[k]*gain1,
					vhat=((double*)model.cpu_adam_v->data)[k]*gain2;
				((double*)model.cpu_params->data)[k]-=lr*mhat/(sqrt(vhat)+epsilon);
			}
		}

		loss=255*sqrt(loss);
		av_loss+=loss;
		++nbatches;

		printf("%d/%d = %5.2lf%% RMSE %16.12lf\t\t\r", ki+1, (int)filenames->count, 100.*(ki+1)/filenames->count, loss);
	}

	save_model(&model, 0, av_loss);
	

	//test
	array_free(&filenames);
	filenames=get_filenames(model.testpath->data, extensions, COUNTOF(extensions));
	ASSERT_MSG(filenames->count, "No images found in test path \'%s\'", (char*)model.testpath->data);

	//change batch size to filenames->count
#if 0
	ASSERT_MSG(filenames->count<=64, "Test dataset has too many samples (max is 64)");
	for(int ki=(int)model.instructions->count-1;ki>=0;--ki)
	{
		Instruction *inst=(Instruction*)array_at(&model.instructions, ki);
		switch(inst->op)
		{
		case OP_CC2:
			break;
		case OP_CONV2:
			break;
		case OP_LRELU:
			break;
		case OP_RELU:
			break;
		case OP_QUANTIZER:
			break;
		case OP_MSE:
			break;
		case OP_MS_SSIM:
			break;
		}
	}
#endif
	ResultCtx res={0};
	res.nsaves=1;//for each sample: {savedBuffers..., lossDiff}
	for(int kb=0;kb<model.buffers->count;++kb)
	{
		Buffer *buf=(Buffer*)array_at(&model.buffers, kb);
		res.nsaves+=buf->type==BUF_NORMAL_SAVED;
	}
	int bsize=model.input_shape[0];
	res.bcountx=(int)ceil(sqrt(bsize));
	res.bcounty=(bsize+res.bcountx-1)/res.bcountx;
	res.bsizex=model.input_shape[3];
	res.bsizey=model.input_shape[2];
	res.size=res.nsaves*res.bcounty*res.bcountx*res.bsizey*res.bsizex;
	res.rgb=(int*)malloc(res.size*sizeof(int));
	int black=0xFF000000;
	memfill(res.rgb, &black, res.size*sizeof(int), sizeof(int));

	load_data((char*)model.trainpath, filenames, &ki, &kblock, &model);
	
	timestamp1=time_ms();
	if(using_gpu)
		loss=forward_gpu(&model, 1, 0, qlevels, &cpu_indices, gpu_indices, &res);
	else
		loss=forward_cpu(&model, 1, &res);
	timestamp2=time_ms();
	
	loss=255*sqrt(loss);
	double psnr=20*log10(255/loss);
	acme_strftime(g_buf, G_BUF_SIZE, (timestamp2-timestamp1)/1000);
	printf("Test RMSE %16.12lf PSNR %13.9lf %10lf ms %s\n", loss, psnr, timestamp2-timestamp1, g_buf);

	//save result
	result_save(&res, loss, 0);

	free_model(&model);
#endif

	//example hardcoded 1D convnet
#if 0
	//initialization
	init_buf(filt1, sizeK, 1);
	init_buf(bias1, 1, 1);
	init_buf(filt2, sizeK, 1);
	init_buf(bias2, 1, 1);
	init_buf(filt3, sizeK, 1);
	init_buf(bias3, 1, 1);

	int nepochs=100000;
	double lr=0.01,
		beta1=0.94, beta2=0.9878, epsilon=1e-8,
		beta1_t=1, beta2_t=1;
	double loss=0;
	long long _t1=__rdtsc();
	for(int it=0;it<nepochs;++it)
	{
		//forward
		cc1d(x, filt1, *bias1, net1, sizeN, sizeK);		//		net1 = cc1d(x, filt1, bias1)
		LeakyReLU(net1, t1, sizeN);						//t1 = act(net1)
		cc1d(t1, filt2, *bias2, net2, sizeN, sizeK);	//		net2 = cc1d(t1, filt2, bias2)
		LeakyReLU(net2, t2, sizeN);						//t2 = act(net2)
		cc1d(t2, filt3, *bias3, net3, sizeN, sizeK);	//		net3 = cc1d(t2, filt3, bias3)
		LeakyReLU(net3, xh, sizeN);						//xh = act(net3)

		subtract(xh, x, diff, sizeN);					//diff = xh-x
		loss=magsq(diff, sizeN)/2*sizeN;				//L = MSE = 1/(2N) sum(sq(diff))

		//loss=0;
		//for(int k=0;k<sizeN;++k)
		//	loss+=diff[k]*diff[k];
		//loss/=2*sizeN;

		//emul(diff, diff, diff2, sizeN);
		//loss=sum_1d(diff2, sizeN)/(2*sizeN);


		//backward
		LeakyReLU_d(net3, dL_dnet3, sizeN);
		emul(diff, dL_dnet3, dL_dnet3, sizeN);
		gmul(dL_dnet3, 1./sizeN, dL_dnet3, sizeN);			//dL/dnet3 = 1/N * diff .* act'(net3)		1/N affects learning rate, doesn't matter for adam?

		//net3 is a function of t2, filt3 & bias3
		conv1d(dL_dnet3, filt3, 0, dL_dt2, sizeN, sizeK);	//dL/dt2 = conv1d(dL/dnet3, filt3, 0)
		int P=(sizeK-1)>>1;
		for(int k=0;k<sizeK;++k)							//dL/dfilt3 = dL/dnet3 dot ( t2[-1:N-1-1], t2[0:N-1], t2[1:N-1+1] )
			dL_dfilt3[k]=dot_sh1d(dL_dnet3, t2, sizeN, k-P);
		*dL_dbias3=sum_1d(dL_dnet3, sizeN);					//dL/dbias3 = sum(dL/dnet3)

		//dt2 is a function of net2
		LeakyReLU_d(net2, dL_dnet2, sizeN);
		emul(dL_dt2, dL_dnet2, dL_dnet2, sizeN);			//dL/dnet2 = dL/dt2 .* act'(net2)

		//net2 is a function of t1, filt2 & bias2
		conv1d(dL_dnet2, filt2, 0, dL_dt1, sizeN, sizeK);	//dL/dt1 = conv1d(dL/dnet2, filt2, 0)
		P=(sizeK-1)>>1;
		for(int k=0;k<sizeK;++k)							//dL/dfilt2 = dL/dnet2 dot ( t1[-1:N-1-1], t1[0:N-1], t1[1:N-1+1] )
			dL_dfilt2[k]=dot_sh1d(dL_dnet2, t1, sizeN, k-P);
		*dL_dbias2=sum_1d(dL_dnet2, sizeN);					//dL/dbias2 = sum(dL/dnet2)

		//dt1 is a function of net1
		LeakyReLU_d(net1, dL_dnet1, sizeN);
		emul(dL_dt1, dL_dnet1, dL_dnet1, sizeN);			//dL/dnet1 = dL/dt1 .* act'(net1)

		//net1 is a function of x, filt1 & bias1
		//conv1d(dL_dnet1, filt1, 0, dL_dx, sizeN, sizeK);	//dL/dx = conv1d(dL/dnet1, filt1, 0)	NOT NEEDED
		P=(sizeK-1)>>1;
		for(int k=0;k<sizeK;++k)							//dL/dfilt1 = dL/dnet1 dot ( x[-1:N-1-1], x[0:N-1], x[1:N-1+1] )
			dL_dfilt1[k]=dot_sh1d(dL_dnet1, x, sizeN, k-P);
		*dL_dbias1=sum_1d(dL_dnet1, sizeN);					//dL/dbias1 = sum(dL/dnet1)


		//adam optimizer
		mix_inplace(adam_m, gradient, beta1, nParams);
		sq_inplace(gradient, nParams);
		mix_inplace(adam_v, gradient, beta2, nParams);

		beta1_t*=beta1;
		beta2_t*=beta2;
		double gain1=1/(1-beta1_t), gain2=1/(1-beta2_t);
		for(int k=0;k<nParams;++k)//https://optimization.cbe.cornell.edu/index.php?title=Adam
		{
			double mhat=adam_m[k]*gain1, vhat=adam_v[k]*gain2;
			params[k]-=lr*mhat/(sqrt(vhat)+epsilon);
		}

		//printf("RMSE=%lf\n", sqrt(loss));
	}
	long long _t2=__rdtsc();
	printf("%d epochs: RMSE=%lf\n%lf cycles/epoch\n", nepochs, sqrt(loss), (double)(_t2-_t1)/nepochs);
	print_1d(x, sizeN, "x: ");
	print_1d(xh, sizeN, "xhat: ");
	//print_1d(diff, sizeN, "xhat-x: ");
#endif

	//OpenCL test
#if 0
	int error;

	printf("ACME-ML\n");
	oclw_loadAPI();

	error=p_clGetPlatformIDs(0, 0, &nplatforms);		CL_CHECK(error);
	ASSERT_MSG(nplatforms, "No OpenCL platforms");
	
	platforms=(cl_platform_id*)malloc(nplatforms*sizeof(cl_platform_id));
	error=p_clGetPlatformIDs(nplatforms, platforms, 0);	CL_CHECK(error);

	error=p_clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, 0, &ndevices);	CL_CHECK(error);
	ASSERT_MSG(ndevices, "No OpenCL devices");

	devices=(cl_device_id*)malloc(ndevices*sizeof(cl_device_id));
	error=p_clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, ndevices, devices, 0);	CL_CHECK(error);
	
	//get info
	size_t retlen=0;
	error=p_clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, G_BUF_SIZE, g_buf, &retlen);CL_CHECK(error);
	printf("OpenCL platform: %s\n", g_buf);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_NAME, G_BUF_SIZE, g_buf, &retlen);	CL_CHECK(error);
	printf("Device: %s\n", g_buf);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, G_BUF_SIZE, g_buf, &retlen);	CL_CHECK(error);
	printf("Vendor: %s\n", g_buf);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxlocalsize, &retlen);	CL_CHECK(error);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE = %d\n", (int)maxlocalsize);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_ADDRESS_BITS, sizeof(size_t), &maxlocalsize, &retlen);	CL_CHECK(error);
	printf("CL_DEVICE_ADDRESS_BITS = %d\n", (int)maxlocalsize);
	error=p_clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, G_BUF_SIZE, g_buf, &retlen);	CL_CHECK(error);
	for(int k=0;k<retlen;++k)
		if(g_buf[k]==' ')
			g_buf[k]='\n';
	printf("Extensions:\n%s\n", g_buf);
	//int cl_gl_interop=strstr(g_buf, "cl_khr_gl_sharing")!=0;
	//if(!cl_gl_interop)
	//	printf("\n\tcl_khr_gl_sharing not supported\n\n");
	
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
	context=p_clCreateContext(properties, 1, &devices[0], nullptr, nullptr, &error);	CL_CHECK(error);
#else
	context=p_clCreateContext(0, ndevices, devices, 0, 0, &error);	CL_CHECK(error);
#endif
	commandqueue=p_clCreateCommandQueue(context, devices[0], 0, &error);	CL_CHECK(error);
#endif

	printf("Done.\n");
	pause();
	return 0;
}