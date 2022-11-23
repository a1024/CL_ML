#ifndef __OPEN_CL__
#include<math.h>
#include<stdio.h>
#define __kernel
#define __global
#define __constant
#define kernel
#define global
#define constant
#define get_global_id(...)		(__VA_ARGS__)
#define get_global_size(...)	(__VA_ARGS__)
#define min(...)				(__VA_ARGS__)
#define max(...)				(__VA_ARGS__)
#define mix(...)				(__VA_ARGS__)
#define clamp(...)				(__VA_ARGS__)
#define as_float(...)			(__VA_ARGS__)
#define as_int(...)				(__VA_ARGS__)
#endif

//	#define DEBUG_KERNELS

	#define		CONST		__constant
//	#define		CONST		__global const
#ifdef FIXED_PREC
typedef int DataType; 
#define		MUL(A, B)		((long)(A)*(B)>>16)
#define		ONE_PERCENT		655//round(0x10000/100)
#define		SQRT(X)			(int)(sqrt((float)(X)/0x10000)*0x10000)
#define		DIV(N, D)		(int)(((long)(N)<<16)/(D))
#define		ZERO			0
#define		ONE				0x10000
#define		MIX(A, B, X)	((A)+(((B)-(A))*(X)>>16))
int			ROUND(int x)
{
	int neg=x>>31&1;

	x^=-neg;//remove sign
	x+=neg;

	x+=0x8000;//sgn(x)*trunc(abs(x)+0.5)
	x&=0xFFFF0000;

	x^=-neg;//restore sign
	x+=neg;

	return x;
}
#else
typedef float DataType;
#define		MUL(A, B)		((A)*(B))
#define		ONE_PERCENT		0.01f
#define		SQRT			sqrt
#define		DIV(N, D)		(N/D)
#define		ZERO			0.f
#define		ONE				1.f
#define		MIX				mix
#define		ROUND			round
#endif

typedef struct ConvInfoStruct//68 bytes
{	//	0	1	2	3	4	5	6	7		8		9			10			11	12	13		14		15		16
	int B,	Ci,	Hi,	Wi,	Co,	Kh,	Kw, xpad,	ypad,	xstride,	ystride,	Ho,	Wo, weight, bias,	epoch,	qlevels;//TODO support stride
} ConvInfo;
//typedef enum CC2IdxEnum
//{
//	CC2_IB, CC2_IC, CC2_IH, CC2_IW,
//	CC2_FO, CC2_FI, CC2_FH, CC2_FW,
//	CC2_OB, CC2_OC, CC2_OH, CC2_OW,
//	CC2_XP, CC2_YP, CC2_XS, CC2_YS,
//} CC2Idx;
__kernel void cc2d			(CONST int *indices, __global const DataType *in, __global const DataType *params, __global DataType *out)//for each output value from [B,Co,Ho,Wo]
{
	int idx=get_global_id(0), idx2=idx;
	CONST ConvInfo *info=(CONST ConvInfo*)indices;
	__global DataType const *filt, *inch;
	int kres=info->Kw*info->Kh, ires=info->Wi*info->Hi;
	//int B=iidx->B,
	//	Ci=iidx->C, Hi=iidx->H, Wi=iidx->W,
	//	Kh=fidx->Kh, Kw=fidx->Kw,
	//	Co=oidx->C, Ho=oidx->H, Wo=oidx->W,
	//	xpad=cidx->xpad, ypad=cidx->ypad, xstride=cidx->xstride, ystride=cidx->ystride,
	//	kres=Kw*Kh, ires=Wi*Hi;
	//int B=indices[CC2_IB],
	//	Ci=indices[CC2_IC], Hi=indices[CC2_IH], Wi=indices[CC2_IW],
	//	Kh=indices[CC2_FH], Kw=indices[CC2_FW],
	//	Co=indices[CC2_OC], Ho=indices[CC2_OH], Wo=indices[CC2_OW],
	//	xpad=indices[CC2_XP], ypad=indices[CC2_YP], xstride=indices[CC2_XS], ystride=indices[CC2_YS],
	//	kres=Kw*Kh, ires=Wi*Hi;
	int kb, koc, koy, kox;
	kox=idx2%info->Wo, idx2/=info->Wo;
	koy=idx2%info->Ho, idx2/=info->Ho;
	koc=idx2%info->Co, idx2/=info->Co;
	kb=idx2;
#ifdef DEBUG_KERNELS
	printf("[%d] %d %d %d %d\n", idx, kb, koc, koy, kox);
#endif
	DataType sum=params[info->bias+koc];
	for(int kic=0;kic<info->Ci;++kic)//for each input channel
	{
		filt=params+info->weight+kres*(info->Ci*koc+kic);
		inch=in+ires*(info->Ci*kb+kic);
//#ifdef DEBUG_KERNELS
//		if(!idx)
//			printf("glob size %d\n%g %g\n", get_global_size(0), params[0], params[9]);
//			//printf("offsets filt %d in %d\n", info->weight+kres*(info->Ci*koc+kic), ires*(info->Ci*kb+kic));
//			//printf("vertically %d to %d\nhorizontally %d to %d\n", koy-info->ypad+0, koy-info->ypad+info->Kh-1, kox-info->xpad+0, kox-info->xpad+info->Kw-1);
//#endif
		for(int kky=0;kky<info->Kh;++kky)//for filter height
		{
			int kiy=koy-info->ypad+kky;
			if((unsigned)kiy<(unsigned)info->Hi)
			{
				for(int kkx=0;kkx<info->Kw;++kkx)//for filter width
				{
					int kix=kox-info->xpad+kkx;
					if((unsigned)kix<(unsigned)info->Wi)
					{
						DataType prod=MUL(filt[info->Kw*kky+kkx], inch[info->Wi*kiy+kix]);
						//if(isfinite(prod))
							sum+=prod;
						//else if(!idx)
						//	printf("%08X\n", as_int(prod));
#ifdef DEBUG_KERNELS
						if(idx==1)
							printf("mul [%d,%d] %g and [%d,%d] %g sum %g\n", kky, kkx, filt[info->Kw*kky+kkx], kiy, kix, inch[info->Wi*kiy+kix], sum);
#endif
					}
				}
			}
		}
	}
#if 0
	if(!idx)
	{
		//printf("%lf\t%lf\t%lf\t%lf\n", in[0], in[1], in[2], in[3]);
		//printf("%lf\t%lf\t%lf\t%lf\n", in[4], in[5], in[6], in[7]);
		//printf("%lf\t%lf\t%lf\t%lf\n", in[8], in[9], in[10], in[11]);
		//printf("%lf\t%lf\t%lf\t%lf\n", in[12], in[13], in[14], in[15]);
	//	printf("%lld\n", sizeof(*out));		//4
	//	printf("%lld\n", sizeof(sum));		//4
	//	printf("%lld\n", sizeof(void*));	//8
	//	printf("%lld\n", sizeof(int));		//4
	//	printf("%lld\n", sizeof(long));		//8
	//	//printf("%lld\n", sizeof(long long));//16, not supported
	//	printf("%lld\n", sizeof(float));	//4
	//	printf("%lld\n", sizeof(double));	//8
	//	printf("CONV %lf, indices %d %d %d %d %d %d %d\n", (double)sum, indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6]);
		//for(int k=0;k<17;++k)
		//	printf("%d ", indices[k]);
	}
#endif
	//out[Wo*(Ho*(Co*kb+koc)+koy)+kox]

	//if(isfinite(sum))
		out[idx]=sum;
	//else if(!idx)
	//	printf("%08X\n", as_int(sum));
}
__kernel void cc2d_grad_in	(CONST int *indices, __global const DataType *dL_dnet, __global const DataType *params, __global DataType *dL_din)//for each input value from [B,Ci,Hi,Wi]
{
	int idx=get_global_id(0), idx2=idx;
	CONST ConvInfo *info=(CONST ConvInfo*)indices;
	__global DataType const *filt, *outch;
	int kres=info->Kw*info->Kh, ores=info->Wo*info->Ho;
	int kb, kic, kiy, kix;
	kix=idx2%info->Wi, idx2/=info->Wi;
	kiy=idx2%info->Hi, idx2/=info->Hi;
	kic=idx2%info->Ci, idx2/=info->Ci;
	kb=idx2;
	DataType sum=0;
	for(int koc=0;koc<info->Co;++koc)//for each output channel
	{
		filt=params+info->weight+kres*(info->Ci*koc+kic);
		outch=dL_dnet+ores*(info->Co*kb+koc);
		for(int kky=0;kky<info->Kh;++kky)//for filt height
		{
			int ky2=kiy-info->ypad+kky;
			if((unsigned)ky2<(unsigned)info->Ho)
			{
				for(int kkx=0;kkx<info->Kw;++kkx)//for filt width
				{
					int kx2=kix-info->xpad+kkx;
					if((unsigned)kx2<(unsigned)info->Wo)
						sum+=MUL(filt[info->Kw*(info->Kh-1-kky)+info->Kw-1-kkx], outch[info->Wo*ky2+kx2]);
					//	sum+=MUL(filt[info->Kw*(info->Kh*(info->Ci*koc+kic)+info->Kh-1-kky)+info->Kw-1-kkx], dL_dnet[info->Wo*(info->Ho*(info->Co*kb+koc)+ky2)+kx2]);
				}
			}
		}
	}
	dL_din[idx]=sum;
}
__kernel void cc2d_grad_filt(CONST int *indices, __global const DataType *dL_dnet, __global const DataType *in, __global DataType *grad_dL_dfilt)//for each weight from [Co,Ci,Kh,Kw]
{
	int idx=get_global_id(0), idx2=idx;
	CONST ConvInfo *info=(CONST ConvInfo*)indices;
	__global DataType const *netch, *outch;
	int ores=info->Wo*info->Ho, ires=info->Wi*info->Hi;
	int koc, kic, kky, kkx;
	kkx=idx2%info->Kw, idx2/=info->Kw;
	kky=idx2%info->Kh, idx2/=info->Kh;
	kic=idx2%info->Ci, idx2/=info->Ci;
	koc=idx2;
	DataType sum=0;
	for(int kb=0;kb<info->B;++kb)//for each sample in batch
	{
		netch=dL_dnet+ores*(info->Co*kb+koc);
		outch=in+ires*(info->Ci*kb+kic);
		for(int koy=0;koy<info->Ho;++koy)//for output height
		{
			int kiy=koy-info->ypad+kky;
			if((unsigned)kiy<(unsigned)info->Hi)
			{
				for(int kox=0;kox<info->Wo;++kox)//for output width
				{
					int kix=kox-info->xpad+kkx;
					if((unsigned)kix<(unsigned)info->Wi)
						sum+=MUL(netch[info->Wo*koy+kox], outch[info->Wi*kiy+kix]);
					//	sum+=MUL(dL_dnet[info->Wo*(info->Ho*(info->Co*kb+koc)+koy)+kox], in[info->Wi*(info->Hi*(info->Ci*kb+kic)+kiy)+kix]);
				}
			}
		}
	}
	grad_dL_dfilt[info->weight+idx]=sum;

	//if(idx<100)//DEBUG
	//	printf("cc2d_grad_filt [[%d]] dL_dfilt %g\n", idx, grad_dL_dfilt[idx]);
}
__kernel void cc2d_grad_bias(CONST int *indices, __global const DataType *dL_dnet, __global DataType *grad_dL_dbias)//for each bias from [Co]
{
	int idx=get_global_id(0);
	CONST ConvInfo *info=(CONST ConvInfo*)indices;
	__global DataType const *src;
	//just sum dL_dnet over [B, -, Ho, Wo]
	int ores=info->Wo*info->Ho;
	DataType sum=0;
	for(int kb=0;kb<info->B;++kb)
	{
		src=dL_dnet+ores*(info->Co*kb+idx);
		for(int k=0;k<ores;++k)
			sum+=src[k];
		//for(int ky=0;ky<info->Ho;++ky)
		//{
		//	for(int kx=0;kx<info->Wo;++kx)
		//		sum+=src[info->Wo*ky+kx];
		//}
	}
	grad_dL_dbias[info->bias+idx]=sum;
}

//element-wise operations (eg: nonlinearity, quantizer) use same indices array as cc/conv, and operate on [B,Co,Ho,Wo]
__kernel void lrelu			(CONST int *indices, __global const DataType *in, __global DataType *out)//for each [B,Co,Ho,Wo]
{
	int idx=get_global_id(0);
	DataType x=in[idx];
	out[idx]=x<0?MUL(ONE_PERCENT, x):x;
}
__kernel void lrelu_grad	(CONST int *indices, __global const DataType *dL_dout, __global const DataType *in, __global DataType *dL_din)//for each [B,Co,Ho,Wo]
{
	int idx=get_global_id(0);
	if(in[idx]<0)//dL_din = dL_dout .* act'(in)
		dL_din[idx]=MUL(ONE_PERCENT, dL_dout[idx]);
	else
		dL_din[idx]=dL_dout[idx];

	//if(idx<100)//DEBUG
	//	printf("lrelu_grad [[%d]] dL_dout %g  in %g -> dL_din %g\n", idx, dL_dout[idx], in[idx], dL_din[idx]);
}

__kernel void relu			(CONST int *indices, __global const DataType *in, __global DataType *out)//for each [B,Co,Ho,Wo]
{
	int idx=get_global_id(0);
	DataType x=in[idx];
	out[idx]=x<0?0:x;
}
__kernel void relu_grad		(CONST int *indices, __global const DataType *dL_dout, __global const DataType *in, __global DataType *dL_din)//for each [B,Co,Ho,Wo]
{
	int idx=get_global_id(0);//dL_din = dL_dout .* act'(in)
	dL_din[idx]=dL_dout[idx]*(in[idx]>=0);
	//if(in[idx]<0)
	//	dL_din[idx]=0;
	//else
	//	dL_din[idx]=dL_dout[idx];
}

DataType rand(unsigned seed)//between [0, 1]
{
	//https://stackoverflow.com/questions/9912143/how-to-get-a-random-number-in-opencl
	//seed=(seed*0x5DEECE66DL+0xBL)&((1L<<48)-1);
	seed=seed*0xDEECE66D+11;
#ifdef FIXED_PREC
	int x=(seed&0xFFFF)^(seed>>16);
#else
	float x=as_float(seed);
#endif
	return x;
}
__kernel void quantizer_train(CONST int *indices, __global const DataType *in, __global DataType *out)
{
	int idx=get_global_id(0);
	CONST ConvInfo *info=(CONST ConvInfo*)indices;
	DataType x=in[idx]+rand(idx*info->epoch)/info->qlevels;//should work with float and fixed
	out[idx]=clamp(x, ZERO, ONE);

	//if(idx<50)//DEBUG
	//	printf("[[%d]] %g clamp %g\n", idx, x, out[idx]);
}
__kernel void quantizer_grad(CONST int *indices, __global const DataType *dL_dout, __global const DataType *in, __global DataType *dL_din)
{
	int idx=get_global_id(0);//dL_din = dL_dout .* rect(in)

	dL_din[idx]=MUL(dL_dout[idx], (in[idx]>=0)-(in[idx]>1));

	//DataType x=dL_dout[idx], y=in[idx];
	//if(y<ZERO)
	//	x=ZERO;
	//if(y>ONE)
	//	x=ZERO;
	//dL_din[idx]=x;

	//if(idx<50)//DEBUG
	//	printf("[[%d]] dL_dout %g in %g\n", idx, dL_dout[idx], in[idx]);
}
__kernel void quantizer_test(CONST int *indices, __global const DataType *in, __global DataType *out)
{
	int idx=get_global_id(0);
	CONST ConvInfo *info=(CONST ConvInfo*)indices;
	DataType x=in[idx];
	x*=info->qlevels;
	x=ROUND(x);
	x/=info->qlevels;
	out[idx]=clamp(x, ZERO, ONE);
}

__kernel void loss_MSE		(CONST int *indices, __global const DataType *s1, __global const DataType *s2, __global DataType *diff)
{
	int idx=get_global_id(0);
	CONST ConvInfo *info=(CONST ConvInfo*)indices;

	//if(idx<50)//DEBUG
	//	printf("[[%d]] xhat %g x %g\n", idx, s1[idx], s2[idx]);//

	diff[idx]=s1[idx]-s2[idx];
}

typedef struct AdamInfoStruct
{
	DataType lr, beta1, beta2, epsilon, gain1, gain2;//gain[i] = 1/(1-pow(beta[i], epoch))
} AdamInfo;
__kernel void opt_adam		(CONST DataType *betas, __global const DataType *grad, __global DataType *adam_m, __global DataType *adam_v, __global DataType *params)//for each learnable parameter
{
	int idx=get_global_id(0);
	CONST AdamInfo *info=(CONST AdamInfo*)betas;

	//params[idx]-=MUL(info->lr, grad[idx]);//SGD

	DataType g2, mhat, vhat, change;
	adam_m[idx]=MIX(adam_m[idx], grad[idx], info->beta1);
	g2=MUL(grad[idx], grad[idx]);
	adam_v[idx]=MIX(adam_v[idx], g2, info->beta2),
	mhat=MUL(adam_m[idx], info->gain1);
	vhat=MUL(adam_v[idx], info->gain2);
	change=DIV(MUL(info->lr, mhat), (SQRT(vhat)+info->epsilon));
	params[idx]-=change;

	//if(idx<50)//DEBUG
	//	printf("[[%d]] param %g lr %g grad %g change %g\n", idx, params[idx], info->lr, grad[idx], change);//
}