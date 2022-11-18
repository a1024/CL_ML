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

#if 0
unsigned nplatforms=0, ndevices=0;
cl_platform_id *platforms=0;
cl_device_id *devices=0;
size_t maxlocalsize=0;
cl_context context=0;
cl_command_queue commandqueue=0;
cl_program program=0;
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
void mix_inplace(double *dst, const double *src, double x, int N)//dst = x*dst + (1-x)*src
{
	for(int k=0;k<N;++k)
		dst[k]=src[k]+(dst[k]-src[k])*x;
}
void sq_inplace(double *x, int N)
{
	for(int k=0;k<N;++k)
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
typedef enum AugmentationTypeEnum
{
	AUG_BLOCK,		//image is padded then partitioned to HxW blocks
	AUG_RANDCROP,	//random crop
	AUG_STRETCH,	//image is stretched to a HxW block
} AugmentationType;
typedef enum InstTypeEnum
{
	//FWD: x[C,H,W], filt[Co,Ci,K,K], bias[Co] -> net[C,H,W]
	//BWD: dL_dnet[C,H,W], filt[Co,Ci,K,K] -> dL_dx[C,H,W], dL_dfilt[Co,Ci,K,K], dL_dbias[Co]
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
	int fwd_args[3], fwd_result, bwd_args[2], bwd_results[3],//indices
		info[4];//[xpad, ypad, xstride, ystride], or [quantization_nlevels]
} Instruction;
typedef enum BufferTypeEnum
{
	//use buf->data
	BUF_NORMAL,

	//use model->params->data+buf->offset*sizeof(double)
	BUF_PARAM, BUF_GRAD,
} BufferType;
typedef struct BufferStruct
{
	int shape[4];//[B, C, H, W], or [Cout, Cin, Ky, Kx]
	BufferType type;
	union
	{
		size_t offset;
		double *data;
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
		result;//index in buffers
	ArrayHandle
		src,	//a string, the source code
		instructions,//array of Instruction
		buffers,//array of Buffer
		params, grad, adam_m, adam_v;//arrays of double (same size)
} Model, *ModelHandle;
void free_buffer(void *p)
{
	Buffer *buf=(Buffer*)p;
	if(buf->type==BUF_NORMAL)
		free(buf->data);
}
void free_model(void *p)
{
	Model *model=(Model*)p;
	array_free(&model->adam_m);
	array_free(&model->adam_v);
	array_free(&model->grad);
	array_free(&model->params);
	array_free(&model->src);
	array_free(&model->instructions);
	array_free(&model->buffers);
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

const char
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
	*ksearch_aug[]={kw_block, kw_randcrop, kw_stretch},
	*ksearch_global[]={kw_cc, kw_conv, kw_save, kw_quantize, kw_loss},
	*ksearch_nl[]={kw_lrelu, kw_relu},
	*ksearch_loss[]={kw_mse, kw_msssim};

void parse_conv(const char *filename, ArrayHandle text, int *ctx, ModelHandle model, ArrayHandle variables, int match)
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
		x->shape[1]=Cin;
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
	inst->bwd_results[0]=dLdx_idx;	//dL_dx
	inst->bwd_results[1]=nbuffers+1;//dL_dfilt
	inst->bwd_results[2]=nbuffers+3;//dL_dbias
	inst->info[0]=inst->info[1]=pad;
	inst->info[2]=inst->info[3]=stride;

	//nonlinearity
	inst[1].fwd_args[0]=nbuffers+4;		//net
	inst[1].fwd_result=nbuffers+6;		//x2
	inst[1].bwd_args[0]=nbuffers+7;		//dL_dx2
	inst[1].bwd_args[1]=nbuffers+6;		//x2
	inst[1].bwd_results[0]=nbuffers+5;	//dL_dnet
}
void parse_save(const char *filename, ArrayHandle text, int *ctx, ModelHandle model, ArrayHandle variables)
{
	Variable *var;
	int start;

	skip_ws(text, ctx);
	start=*ctx;
	get_id(text, ctx);
	int found=0;
	for(int k=0;k<(int)variables->count;++k)
	{
		var=(Variable*)array_at(&variables, k);
		if(var->name->count==*ctx-start&&!memcmp(text->data+start, var->name->data, var->name->count))
		{
			found=1;
			break;
		}
	}
	ASSERT_MSG(!found, "%s(%d): Saved variable can only appear once, \'%.*s\' appeared before", filename, ctx[1], *ctx-start, text->data+start);
	var=(Variable*)ARRAY_APPEND(variables, 0, 1, 1, 0);
	STR_COPY(var->name, text->data+start, *ctx-start);
	if(model->instructions->count)
	{
		Instruction *inst=(Instruction*)array_at(&model->instructions, model->instructions->count-1);
		var->buffer=inst->fwd_result;
	}
	else
		var->buffer=-1;
}
void parse_quantize(const char *filename, ArrayHandle text, int *ctx, ModelHandle model, ArrayHandle variables)
{
	Instruction *inst;
	Buffer *buffer;
	double fval;
	
	ASSERT_MSG(model->instructions->count, "%s(%d): Quantizer cannot be the first operation", filename, *ctx);//because Cin would be unknown
	int first_inst=!model->instructions->count;
	inst=(Instruction*)ARRAY_APPEND(model->instructions, 0, 1, 1, 0);

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
void parse_loss(const char *filename, ArrayHandle text, int *ctx, ModelHandle model, ArrayHandle variables)
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
	ASSERT_MSG(!memcmp(buf1->shape, buf2->shape, sizeof(buf2->shape)), "%s(%d): Loss function expects two buffers of identical shape, got %dx%dx%dx%d & %dx%dx%dx%d", filename, ctx[1],
		buf1->shape[0], buf1->shape[1], buf1->shape[2], buf1->shape[3], buf2->shape[0], buf2->shape[1], buf2->shape[2], buf2->shape[3]);
	memcpy(diff->shape, buf1->shape, sizeof(buf1->shape));
	diff->type=BUF_NORMAL;

	inst->fwd_args[0]=b1;
	inst->fwd_args[1]=b2;
	inst->fwd_result=nbuffers;
	inst->bwd_results[0]=nbuffers;
}
void init_model(ModelHandle model)
{
	//for(int ki=0;ki<(int)model->instructions->count;++ki)//Xavier initialization
	//{
	//	Instruction *inst=(Instruction*)array_at(&model->instructions, ki);
	//}
	double gain=1./RAND_MAX;
	size_t kp=0;
	for(int kb=0;kb<(int)model->buffers->count;++kb)
	{
		Buffer *buf=(Buffer*)array_at(&model->buffers, kb);
		if(buf->type==BUF_PARAM)
		{
			size_t bufsize=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
			for(size_t k0=0;k0<bufsize;++k0)
			{
				double *val=(double*)array_at(&model->params, kp);
				*val=rand()*gain;
			}
		}
	}
}
void parse_model(const char *filename, ModelHandle model)
{
	ArrayHandle text=load_text(filename, 0),
		variables;//array of Variable
	int ctx[3]={0}, match;//idx, lineno, linestart
	double fval=0;
	int ival;
	if(!text)
		LOG_ERROR("Cannot open \'%s\'\n", filename);

	ARRAY_ALLOC(Variable, variables, 0, 0, 0, free_variable);

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

	ARRAY_ALLOC(Instruction, model->instructions, 0, 0, 0, 0);
	ARRAY_ALLOC(Buffer, model->buffers, 0, 0, 0, free_buffer);

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
			parse_save(filename, text, ctx, model, variables);
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

	STR_COPY(model->src, text->data+*ctx, *ctx);

	//allocate buffers
	size_t nParams=0, nGrad=0;
	for(int k=0;k<(int)model->buffers->count;++k)
	{
		Buffer *buf=array_at(&model->buffers, k);
		size_t bufsize=buf->shape[0]*buf->shape[1]*buf->shape[2]*buf->shape[3];
		ASSERT_MSG(bufsize, "%s: Buffer %d size is zero", filename, k);
		switch(buf->type)
		{
		case BUF_NORMAL:
			buf->data=(double*)malloc(bufsize*sizeof(double));
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
	ARRAY_ALLOC(double, model->params, 0, nParams, 0, 0);
	ARRAY_ALLOC(double, model->grad, 0, nGrad, 0, 0);
	ARRAY_ALLOC(double, model->adam_m, 0, nGrad, 0, 0);
	ARRAY_ALLOC(double, model->adam_v, 0, nGrad, 0, 0);

	skip_ws(text, ctx);
	if(!match_kw(text, ctx, (const char**)&kw_weights, 1))
	{
		for(size_t k=0;k<nParams;++k)
		{
			double *param=(double*)array_at(&model->params, k);
			skip_ws(text, ctx);
			parse_number(filename, text, ctx, 16, param);
		}
	}
	else
		init_model(model);

	skip_ws(text, ctx);
	ASSERT_MSG(*ctx==text->count, "%s(%d): Expected end of file", filename, ctx[1]);

	array_free(&variables);
	array_free(&text);
}

int print_hex(char *buf, size_t len, double x)
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
			buf[idx]=digit+(digit<10?'0':'A');
			++idx;
		}
	}
	return idx;
}
void save_model(ModelHandle model, int incremental)
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
		printed=sprintf_s(g_buf, G_BUF_SIZE, "model-%04d%02d%02d-%02d%02d%02d%s.txt", 1900+ts->tm_year, 1+ts->tm_mon, ts->tm_mday, ts->tm_hour%12, ts->tm_min, ts->tm_sec, ts->tm_hour<12?"AM":"PM");
		STR_COPY(filename, g_buf, printed);
	}
	STR_COPY(text, model->src->data, model->src->count);

	size_t kp=0;
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
						for(int k3=0;k3<buf->shape[3];++k3, ++kp)
						{
							double *val=(double*)array_at(&model->params, kp);
							printed=print_hex(g_buf, G_BUF_SIZE, *val);
							STR_APPEND(text, "\t", 1, 1);
							STR_APPEND(text, g_buf, printed, 1);
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
	//	printed=print_hex(g_buf, G_BUF_SIZE, *val);
	//	STR_APPEND(text, "\n", 1, 1);
	//	STR_APPEND(text, g_buf, printed, 1);
	//}

	save_text(filename->data, text->data, text->count);
	array_free(&text);
	array_free(&filename);
}

void addsh2d(double *tensor)
{
}
void cc2d_fwd(Model *model, Instruction *inst)
{
	Buffer
		*bufx=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
		*buff=(Buffer*)array_at(&model->buffers, inst->fwd_args[1]),
		*bufb=(Buffer*)array_at(&model->buffers, inst->fwd_args[2]),
		*buf2=(Buffer*)array_at(&model->buffers, inst->fwd_result);
#define FETCH(PTR, BUFFER)	PTR=BUFFER->type==BUF_NORMAL?BUFFER->data:(BUFFER->type==BUF_PARAM?(double*)array_at(&model->params, BUFFER->offset):(double*)array_at(&model->grad, BUFFER->offset))
	double *data, *filt, *bias, *d2;
	FETCH(data, bufx);
	FETCH(filt, buff);
	FETCH(bias, bufb);
	FETCH(d2, buf2);
#undef	FETCH

	size_t ires=bufx->shape[2]*bufx->shape[3],
		isize=bufx->shape[1]*bufx->shape[2]*bufx->shape[3],
		osize=buf2->shape[1]*buf2->shape[2]*buf2->shape[3],
		fsize=buf2->shape[1]*buf2->shape[2]*buf2->shape[3];
	int ypad=buff->shape[2]>>1, xpad=buff->shape[3]>>1;
	for(int kb=0;kb<buf2->shape[0];++kb)//for each batch
	{
		double *outsample=d2+osize*kb;
		for(int ko=0;ko<buf2->shape[1];++ko)//for output channel
		{
			for(int ky=0;ky<buf2->shape[2];++ky)//for each row
			{
				for(int kx=0;kx<buf2->shape[3];++kx, ++outsample)//for each pixel
				{
					double *insample=data+isize*kb, *kernel=filt+fsize*ko;
					double sum=bias[ko];
					for(int ki=0;ki<buff->shape[1];++ki, insample+=ires)//for each input channel
					{
						for(int ky2=0;ky2<buff->shape[2];++ky2)//for each kernel row
						{
							unsigned yidx=ky+ky2-ypad;
							if((unsigned)yidx<(unsigned)buf2->shape[2])
							{
								double *inrow=insample+bufx->shape[3]*yidx;
								for(int kx2=0;kx2<buff->shape[3];++kx2, ++inrow, ++kernel)//for each kernel value
								{
									unsigned xidx=kx+kx2-xpad;
									if((unsigned)xidx<(unsigned)buf2->shape[3])
										sum+=*kernel**inrow;
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
void quantizer_train_fwd(Model *model, Instruction *inst)
{
	Buffer
		*src=(Buffer*)array_at(&model->buffers, inst->fwd_args[0]),
		*dst=(Buffer*)array_at(&model->buffers, inst->fwd_result);
	ASSERT_MSG(src->type==BUF_NORMAL, "Quantizer cannot be applied to learnable parameters, src->type = %d", src->type);
	ASSERT_MSG(dst->type==BUF_NORMAL, "Quantizer cannot be applied to learnable parameters, src->type = %d", dst->type);
	int size=src->shape[0]*src->shape[1]*src->shape[2]*src->shape[3];
	double gain=1./((double)(inst->info[0]-1)*RAND_MAX);
	for(size_t k=0;k<size;++k)
		dst->data[k]=src->data[k]+gain*(rand()-(RAND_MAX>>1));
}
Model model={0};
#endif

int main(int argc, char **argv)
{
	set_console_buffer_size(120, 2000);
	
	//automatic differentiation in 2D
#if 1
	printf(
		"ACME-ML\t\tbuild %s %s\n", __DATE__, __TIME__);
	if(argc<5||argc>6)
	{
		printf(
			"Usage:\n"
			"  program  datasetPath  epochCount  learningRate  batchSize  [modelName.txt]\n"
			"If model name is not given, will attempt to load the incremental \'model.txt\'\n"
			"\n"
		);
#ifdef _MSC_VER
		pause();
#endif
		return 1;
	}
	int epoch=atoi(argv[2]);
	double lr=atof(argv[3]);
	model.input_shape[0]=atoi(argv[4]);//Batch size
	parse_model(argc==5?"model.txt":argv[5], &model);

	//train
	double loss=0;
	for(int ke=0;ke<epoch;++ke)
	{
		//load data

		//forward
		for(int ki=0;ki<(int)model.instructions->count;++ki)
		{
			Instruction *inst=(Instruction*)array_at(&model.instructions, ki);
			switch(inst->op)
			{
			case OP_CC2:
				cc2d_fwd(&model, inst);
				continue;
			case OP_CONV2:
				LOG_ERROR("Conv2d is not supported yet. Use CC2D.");
				continue;
			case OP_LRELU:
				{
					Buffer
						*src=(Buffer*)array_at(&model.buffers, inst->fwd_args[0]),
						*dst=(Buffer*)array_at(&model.buffers, inst->fwd_result);
					ASSERT_MSG(src->type==BUF_NORMAL, "LReLU cannot be applied to learnable parameters, src->type = %d", src->type);
					ASSERT_MSG(dst->type==BUF_NORMAL, "LReLU cannot be applied to learnable parameters, src->type = %d", dst->type);
					LeakyReLU(src->data, dst->data, src->shape[0]*src->shape[1]*src->shape[2]*src->shape[3]);
				}
				continue;
			case OP_RELU:
				{
					Buffer
						*src=(Buffer*)array_at(&model.buffers, inst->fwd_args[0]),
						*dst=(Buffer*)array_at(&model.buffers, inst->fwd_result);
					ASSERT_MSG(src->type==BUF_NORMAL, "ReLU cannot be applied to learnable parameters, src->type = %d", src->type);
					ASSERT_MSG(dst->type==BUF_NORMAL, "ReLU cannot be applied to learnable parameters, src->type = %d", dst->type);
					ReLU(src->data, dst->data, src->shape[0]*src->shape[1]*src->shape[2]*src->shape[3]);
				}
				continue;
			case OP_QUANTIZER:
				quantizer_train_fwd(&model, inst);
				continue;
			case OP_MSE:
				{
					Buffer
						*s1=(Buffer*)array_at(&model.buffers, inst->fwd_args[0]),
						*s2=(Buffer*)array_at(&model.buffers, inst->fwd_args[1]),
						*dst=(Buffer*)array_at(&model.buffers, inst->fwd_result);
					ASSERT_MSG(s1->type==BUF_NORMAL, "MSE cannot be applied to learnable parameters, src->type = %d", s1->type);
					ASSERT_MSG(s2->type==BUF_NORMAL, "MSE cannot be applied to learnable parameters, src->type = %d", s2->type);
					ASSERT_MSG(dst->type==BUF_NORMAL, "MSE cannot be applied to learnable parameters, src->type = %d", dst->type);
					size_t count=s1->shape[0]*s1->shape[1]*s1->shape[2]*s1->shape[3];
					loss=0;
					for(size_t k=0;k<count;++k)
					{
						dst->data[k]=s1->data[k]-s2->data[k];
						loss+=dst->data[k]*dst->data[k];
					}
					loss/=s1->shape[2]*s1->shape[3]<<1;
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

		//backward
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
			default:
				LOG_ERROR("Unrecognized instruction type %d.", inst->op);
				break;
			}
		}
	}

	save_model(&model, 0);
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