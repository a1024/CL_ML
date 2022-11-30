#pragma once
#ifndef INC_MODEL_H
#define INC_MODEL_H
#ifndef INC_OCL_WRAP_H
#define	CL_TARGET_OPENCL_VERSION	300//120
#include<CL/cl.h>
#endif
#include"array.h"
#ifdef __cplusplus
extern "C"
{
#endif

	
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
} Buffer;
typedef struct ModelStruct
{
	int nepochs;
	AugmentationType aug;
	double lr;
	int input_shape[4],//[B,C,W,H]
		is_test,
		input_idx;//index in buffers
	ArrayHandle
		src,				//string with the model source code, for saving
		trainpath, testpath,//strings with paths to datasets
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
	float *input,
		*gpu_output;
} Model;
	
#ifdef __cplusplus
}
#endif
#endif//INC_MODEL_H
