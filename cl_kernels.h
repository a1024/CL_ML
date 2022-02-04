#ifndef __OPEN_CL__
#include<math.h>
#include<stdio.h>
#define __kernel
#define __global
#define __constant
#define get_global_id(...)	(__VA_ARGS__)
#define max(...)			(__VA_ARGS__)
#endif

enum	IndicesIdx//for indices array
{
	II_Cin, II_Win, II_Hin,
	II_Cout, II_Wout, II_Hout,
	II_logstride, II_Wk, II_Hk,
	II_bufsize,
};
__kernel void zeromem		(__global float *dst)
{
	int idx=get_global_id(0);
	dst[idx]=0;
}
//__kernel void pad_data		(__global float *dst, __global float *src, __constant int *indices)
//{
//	int idx=get_global_id(0);
//	int w1=indices[II_Win], h1=indices[II_Hin], pad=indices[II_Pout], w2=w1+pad*2, h2=h1+pad*2;
//	int kx=idx%w2, ky=idx/w2%h2, kc=idx/(w2*h2);
//	if(kx>=pad&&kx<pad+w1&&ky>=pad&&ky<pad+h1)
//		dst[idx]=src[w1*(h1*kc+ky-pad)+kx-pad];
//	else
//		dst[idx]=0;
//}
//__kernel void res_save	(__global float *dst, __global float *src)
//{
//	int idx=get_global_id(0);
//	dst[idx]=src[idx];
//}
//__kernel void res_save_ds	(__global float *dst, __global float *src, __constant int *indices, __global float *weights)
//{
//	int idx=get_global_id(0);
//	int w1=indices[II_Win], h1=indices[II_Hin], padin=indices[II_Pin],
//		w2=indices[II_Wout], h2=indices[II_Hout], padout=indices[II_Pout];
//	int kxdst=idx%w2, kydst=idx/w2%h2, kc=idx/(w2*h2);
//	int kxsrc=kxdst<<1, kysrc=kydst<<1;
//	kxdst+=padout, kydst+=padout;
//	kxsrc+=padin, kysrc+=padin;
//	dst[w2*(h2*kc+kydst)+kxdst]=weights[kc]*src[w1*(h1*kc+kysrc)+kxsrc]+weights[indices[II_Cout]+kc];
//}
__kernel void res_add		(__global float *src, __global float *dst)
{
	int idx=get_global_id(0);
	dst[idx]+=src[idx];
}
__kernel void relu			(__global float *dst)
{
	int idx=get_global_id(0);
	//printf("ReLU %d: %f -> %f\n", idx, dst[idx], max(0.f, dst[idx]));//
	dst[idx]=max(0.f, dst[idx]);
}
__kernel void maxpool		(__global float *src, __global float *dst, __constant int *indices)
{
	int idx=get_global_id(0);
	int w1=indices[II_Win], h1=indices[II_Hin];
	int w2=indices[II_Wout], h2=indices[II_Hout];
	int wk=indices[II_Wk], hk=indices[II_Hk], ksize=wk*hk;
	int kx=idx%w2, ky=idx/w2%h2, kc=idx/(w2*h2);
	//if(!idx)
	//	printf("%d x %d\n", wk, hk);

	__global float *ch=src+w1*h1*kc;
	kx*=wk, ky*=hk;
	float acc=ch[w1*ky+kx];
	//if(idx<100)
	//	printf("%f", acc);
	for(int y=0;y<hk;++y, ++ky)
		for(int x=0;x<wk;++x, ++kx)
			acc=max(acc, ch[w1*ky+kx]);
	dst[idx]=acc;

	//int idx=get_global_id(0);
	//int wk=indices[II_Wk], hk=indices[II_Hk], ksize=wk*hk, kp;
	//
	//__global float *ch=src+ksize*idx;
	//float acc=ch[0];
	//for(kp=1;kp<ksize;++kp)
	//	acc=max(acc, ch[kp]);
	//dst[idx]=acc;
}
__kernel void avpool		(__global float *src, __global float *dst, __constant int *indices)
{
	int idx=get_global_id(0);
	int w1=indices[II_Win], h1=indices[II_Hin];
	int w2=indices[II_Wout], h2=indices[II_Hout];
	int wk=indices[II_Wk], hk=indices[II_Hk], ksize=wk*hk;
	int kx=idx%w2, ky=idx/w2%h2, kc=idx/(w2*h2);
	
	__global float *ch=src+w1*h1*kc;
	float sum=0;
	kx*=wk, ky*=hk;
	for(int y=0;y<hk;++y, ++ky)
		for(int x=0;x<wk;++x, ++kx)
			sum+=ch[w1*ky+kx];
	//printf("%d ", ksize);
	dst[idx]=sum/ksize;
}
__kernel void conv11		(__global float *src, __global float *dst, __constant int *indices, __global float *weights)
{
	int idx=get_global_id(0);
	int nchin=indices[II_Cin],
		win=indices[II_Win], hin=indices[II_Hin], chsize=win*hin,
		wout=indices[II_Wout], hout=indices[II_Hout];
	int xout=idx%wout, yout=idx/wout%hout, kcout=idx/(wout*hout);
	int xin=xout<<indices[II_logstride], yin=yout<<indices[II_logstride], kcin;

	__global float *filt=weights+(nchin+1)*kcout;
	float result=filt[nchin];//bias
	__global float *px=src+win*yin+xin;

	for(kcin=0;kcin<indices[II_Cin];++kcin, px+=chsize)
		result+=filt[kcin]*px[0];
	dst[idx]=result;
	//dst[wout*(hout*kcout+yout)+xout]=result;
}
__kernel void conv_zp		(__global float *src, __global float *dst, __constant int *indices, __global float *weights)
{
	int idx=get_global_id(0);
	int nchin=indices[II_Cin],
		win=indices[II_Win], hin=indices[II_Hin], chsize=win*hin,
		wout=indices[II_Wout], hout=indices[II_Hout],
		wk=indices[II_Wk], hk=indices[II_Hk];
	int xout=idx%wout, yout=idx/wout%hout, kcout=idx/(wout*hout);
	int xin=xout<<indices[II_logstride], yin=yout<<indices[II_logstride], kcin;
	int xsstart, xsend, ysstart, ysend;
	int xfstart, xfend, yfstart, yfend;

	int convsize=nchin*wk*hk;
	__global float *filt=weights+(convsize+1)*kcout;
	float result=filt[convsize];//bias
	__global float *px=src;

	xsstart	=xin-(wk>>1);	if(xsstart<0)xfstart=-xsstart, xsstart=0;		else xfstart=0;
	xsend	=xin+(wk>>1);	if(xsend>win)xfend=wk-(xsend-win), xsend=win;	else xfend=wk;
	ysstart	=yin-(hk>>1);	if(ysstart<0)yfstart=-ysstart, ysstart=0;		else yfstart=0;
	ysend	=yin+(hk>>1);	if(ysend>hin)yfend=hk-(ysend-hin), ysend=hin;	else yfend=hk;
	//if(idx==0)//
	//{
	//	printf("\n%d~%d, %d~%d\n", xsstart, xsend, ysstart, ysend);
	//	for(int k=0;k<II_bufsize;++k)//
	//		printf("%d ", indices[k]);
	//	printf("%d", convsize);
	//	printf("First %dx%dx%d+1=%d filter GPU:", nchin, wk, hk, convsize+1);
	//	for(int k=0;k<convsize+1;++k)
	//		printf("%f ", filt[k]);
	//}
	for(kcin=0;kcin<nchin;++kcin, px+=chsize, filt+=wk*hk)
	{
		for(int y=ysstart, ky=yfstart;y<ysend;++y, ++ky)
		{
			for(int x=xsstart, kx=xfstart;x<xsend;++x, ++kx)
			{
				//if(!idx)
				//	printf("(c=%d, %d, %d)", kcin, kx, ky);
				result+=filt[wk*ky+kx]*px[win*y+x];
			}
		}
	}
	//if(!indices[II_Cout]&&idx<100)
	//	printf("row(%f, %f, %f) -> %f ", src[win*ysstart+xsstart], src[win*yin+xin], src[win*(ysend-1)+xsend-1], result);
	dst[idx]=result;
	//dst[wout*(hout*kcout+yout)+xout]=result;
}
__kernel void linear_fc		(__global float *src, __global float *dst, __constant int *indices, __global float *weights)
{
	int idx=get_global_id(0);
	int ni=indices[II_Cin], no=indices[II_Cout];

	__global float *node=weights+ni*idx;
	float sum=weights[ni*no+idx];//bias
	for(int k=0;k<ni;++k)
		sum+=node[k]*src[k];
	//if(idx<100)
	//	printf("%f ", sum);
	dst[idx]=sum;
}
#if 0
__kernel void conv_n11		(__global float *dst, __global float *src, __constant int *indices, __global float *weights)
{
	int idx=get_global_id(0);
	int kx=idx%indices[II_Wout], ky=idx/indices[II_Wout]%indices[II_Hout], kcout=idx/(indices[II_Wout]*indices[II_Hout]);
	int xin=1+(kx<<indices[II_logstride]), yin=1+(ky<<indices[II_logstride]);
	int xout=indices[II_Pout]+kx, yout=indices[II_Pout]+ky;
	int kcin;

	__global float *filt=weights+indices[II_Cin]*2*kcout;
	float result=filt[1];
	const int srcW=indices[II_Win]+2, srcinc=srcW*(indices[II_Hin]+2);
	__global float *sch=src, *row;

	for(kcin=0;kcin<indices[II_Cin];++kcin, sch+=srcinc)
	{
		row=sch+srcW*yin;
		result+=filt[0]*row[xin-1];
	}
	dst[indices[II_Wout]*yout+xout]=result;
}
__kernel void conv_n33_zp	(__global float *dst, __global float *src, __constant int *indices, __global float *weights)
{
	int idx=get_global_id(0);
	int w1=indices[II_Win], h1=indices[II_Hin], w2=indices[II_Wout], h2=indices[II_Hout];
	int kx=idx%w2, ky=idx/w2%h2, kcout=idx/(w2*h2);
	int xin=1+(kx<<indices[II_logstride]), yin=1+(ky<<indices[II_logstride]);
	int xout=indices[II_Pout]+kx, yout=indices[II_Pout]+ky;
	int kcin, k;

	__global float *filt=weights+indices[II_Cin]*10*kcout;
	float result=filt[9];
	const int srcW=w1+2, srcinc=srcW*(h1+2);
	__global float *sch=src, *row;

	for(kcin=0;kcin<indices[II_Cin];++kcin, sch+=srcinc)
	{
		row=sch+srcW*yin;
		for(ky=0, k=0;ky<3;++ky)
		{
			for(kx=0;kx<3;++kx, ++k)
				result+=filt[k]*row[xin+k-1];
			row+=srcW;
		}
	}
	dst[w2*(h2*kcout+yout)+xout]=result;
}
__kernel void conv_n77_zp	(__global float *dst, __global float *src, __constant int *indices, __global float *weights)
{
	int idx=get_global_id(0);
	int w1=indices[II_Win], h1=indices[II_Hin], w2=indices[II_Wout], h2=indices[II_Hout];
	int kx=idx%w2, ky=idx/w2%h2, kcout=idx/(w2*h2);
	int xin=3+(kx<<indices[II_logstride]), yin=3+(ky<<indices[II_logstride]);
	int xout=indices[II_Pout]+kx, yout=indices[II_Pout]+ky;
	int kcin, k;

	__global float *filt=weights+indices[II_Cin]*50*kcout;
	float result=filt[49];
	const int srcW=w1+6, srcinc=srcW*(h1+6);
	__global float *sch=src, *row;

	//if(!idx)//
	//	for(int k=0;k<9;++k)
	//		printf("%d", indices[k]);
		//printf("%d/%d%%%d = %d", idx, w2, h2, idx/w2%h2);
		//printf("(%d,%d): (%d,%d) -> (%d,%d)", kx, ky, xin, yin, xout, yout);//

	for(kcin=0;kcin<indices[II_Cin];++kcin, sch+=srcinc)
	{
		row=sch+srcW*yin;
		for(ky=0, k=0;ky<7;++ky)
		{
			for(kx=0;kx<7;++kx, ++k)
				result+=filt[k]*row[xin+k-3];
			row+=srcW;
		}
	}
	dst[w2*(h2*kcout+yout)+xout]=result;
}
#endif

#if 0
//	#define		DEBUG_DWT

__kernel void add(__global float *src, __global float *dst)
{
	int idx=get_global_id(0);
	dst[idx]+=src[idx];
}
__kernel void relu(__global float *src, __global float *dst)
{
	int idx=get_global_id(0);
	dst[idx]=(src[idx]+fabs(src[idx]))*0.5f;
}
__kernel void relu_inplace(__global float *data)
{
	int idx=get_global_id(0);
	data[idx]+=fabs(data[idx]);
	data[idx]*=0.5f;
}
__kernel void lrelu_inplace(__global float *data, __global int *gain)
{
	int idx=get_global_id(0);
	data[idx]=max(0.01f*data[idx], data[idx]);
}
__kernel void conv33_const(__global float *src, __global int *dim, __global float *filt, __global float *dst)
{
	int idx=get_global_id(0);
	int kx=idx%dim[0], ky=idx/dim[0];

	//const padding
	int kxm1=kx-(kx>0), kxp1=kx+(kx+1<dim[0]),
		kym1=ky-(ky>0), kyp1=ky+(ky+1<dim[1]);

	kym1*=dim[0], ky*=dim[0], kyp1*=dim[0];
	dst[idx]=
		filt[0]*src[kym1+kxm1]+filt[1]*src[kym1+kx]+filt[2]*src[kym1+kxp1]+
		filt[3]*src[ky  +kxm1]+filt[4]*src[ky  +kx]+filt[5]*src[ky  +kxp1]+
		filt[6]*src[kyp1+kxm1]+filt[7]*src[kyp1+kx]+filt[8]*src[kyp1+kxp1];
}
__kernel void conv33_mirror(__global float *src, __global int *dim, __global float *filt, __global float *dst)
{
	int idx=get_global_id(0);
	int kx=idx%dim[0], ky=idx/dim[0];
	int kxm1=kx-1, kxp1=kx+1, kym1=ky-1, kyp1=ky+1, mask;

	//mirror padding
	mask=kxm1<0, kxm1^=-mask, kxm1+=mask;
	kxp1-=2&-(kxp1>=dim[0]);

	mask=kym1<0, kym1^=-mask, kym1+=mask;
	kyp1-=2&-(kyp1>=dim[1]);

	kym1*=dim[0], ky*=dim[0], kyp1*=dim[0];
	dst[idx]=
		filt[0]*src[kym1+kxm1]+filt[1]*src[kym1+kx]+filt[2]*src[kym1+kxp1]+
		filt[3]*src[ky  +kxm1]+filt[4]*src[ky  +kx]+filt[5]*src[ky  +kxp1]+
		filt[6]*src[kyp1+kxm1]+filt[7]*src[kyp1+kx]+filt[8]*src[kyp1+kxp1];
}
__kernel void conv33_periodic(__global float *src, __global int *dim, __global float *filt, __global float *dst)
{
	int idx=get_global_id(0);
	int kx=idx%dim[0], ky=idx/dim[0];
	int kxm1=kx-1, kxp1=kx+1, kym1=ky-1, kyp1=ky+1;

	//periodic padding
	kxm1+=dim[0]&-(kxm1<0);
	kxp1-=dim[0]&-(kxm1>=dim[0]);

	kym1+=dim[1]&-(kym1<0);
	kyp1-=dim[1]&-(kym1>=dim[1]);

	kym1*=dim[0], ky*=dim[0], kyp1*=dim[0];
	dst[idx]=
		filt[0]*src[kym1+kxm1]+filt[1]*src[kym1+kx]+filt[2]*src[kym1+kxp1]+
		filt[3]*src[ky  +kxm1]+filt[4]*src[ky  +kx]+filt[5]*src[ky  +kxp1]+
		filt[6]*src[kyp1+kxm1]+filt[7]*src[kyp1+kx]+filt[8]*src[kyp1+kxp1];
}

int xm1_mirror(int x)
{
	--x;
	int mask=x<0;
	x^=-mask, x+=mask;
	return x;
}
int xp1_mirror(int x, int size)
{
	++x;
	x-=2&-(x>=size);
	return x;
}
enum DimIdx
{
	DIM_BW, DIM_BH,
	DIM_XSIZE, DIM_YSIZE,
	DIM_NXODD, DIM_NYODD,
	DIM_FILT,
};
enum FiltIdx
{
	FILT_O_HI,
	FILT_E_LO,
	FILT_Q_AMP,
	FILT_Q_LO,
	FILT_Q_HI,
};
__kernel void lift_2D_H_ohi(__global float *data, __global int *dim, __global float *filt)
{
	int bw=dim[DIM_BW], xsize=dim[DIM_XSIZE], nxodd=dim[DIM_NXODD];
	int idx=get_global_id(0);
	int kx=(idx%nxodd<<1)+1, ky=idx/nxodd;
	int kxm1=xm1_mirror(kx), kxp1=xp1_mirror(kx, xsize);

	int kyw=ky*bw;
	data[kyw+kx]+=filt[dim[DIM_FILT]+FILT_O_HI]*(data[kyw+kxm1]+data[kyw+kxp1]);
}
__kernel void lift_2D_H_elo(__global float *data, __global int *dim, __global float *filt)
{
	int bw=dim[DIM_BW], xsize=dim[DIM_XSIZE], nxeven=xsize-dim[DIM_NXODD];
	int idx=get_global_id(0);
	int kx=idx%nxeven<<1, ky=idx/nxeven;
	int kxm1=xm1_mirror(kx), kxp1=xp1_mirror(kx, xsize);

	int kyw=ky*bw;
	data[kyw+kx]+=filt[dim[DIM_FILT]+FILT_E_LO]*(data[kyw+kxm1]+data[kyw+kxp1]);
}
__kernel void lift_2D_V_ohi(__global float *data, __global int *dim, __global float *filt)
{
	int bw=dim[DIM_BW], xsize=dim[DIM_XSIZE], ysize=dim[DIM_YSIZE];
	int idx=get_global_id(0);
	int kx=idx%xsize, ky=(idx/xsize<<1)+1;
	int kym1=xm1_mirror(ky), kyp1=xp1_mirror(ky, ysize);

	data[bw*ky+kx]+=filt[dim[DIM_FILT]]*(data[bw*kym1+kx]+data[bw*kyp1+kx]);
}
__kernel void lift_2D_V_elo(__global float *data, __global int *dim, __global float *filt)
{
	int bw=dim[DIM_BW], xsize=dim[DIM_XSIZE], ysize=dim[DIM_YSIZE];
	int idx=get_global_id(0);
	int kx=idx%xsize, ky=idx/xsize<<1;
	int kym1=xm1_mirror(ky), kyp1=xp1_mirror(ky, ysize);

	data[bw*ky+kx]+=filt[dim[DIM_FILT]+1]*(data[bw*kym1+kx]+data[bw*kyp1+kx]);
}
/*__kernel void lift_2D(__global float *src, __global float *dst, __global int *dim, __global float *filt)
{
	int bw=dim[0], odd=dim[1]&1, xsize=dim[2], ysize=dim[3], nxeven=dim[4], nyeven=dim[5];
	int idx=get_global_id(0);
	int kx=(idx%nxeven<<1)+odd, ky=idx/nxeven<<1;
//#ifdef DEBUG_DWT
//	printf("%d (%d, %d)", idx, kx, ky);
//#endif
	int kxm1, kxp1, kym1, kyp1, mask, kyw=ky*bw;
	if(kx<xsize)
	{
		kxm1=kx-1, kxp1=kx+1;
		kym1=ky-1, kyp1=ky+1;

		//mirror padding
		mask=kxm1<0, kxm1^=-mask, kxm1+=mask,  kxp1-=2&-(kxp1>=xsize);
		mask=kym1<0, kym1^=-mask, kym1+=mask,  kyp1-=2&-(kyp1>=ysize);
#ifdef DEBUG_DWT
		printf("A: %d (%d, %d) (%d->%d, %d->%d): %f %f %f\n", idx, kx, ky, kxm1, kxp1, kym1, kyp1, src[kyw+kx], filt[dim[1]], src[kyw+kx] + filt[dim[1]]*(src[kyw+kxm1]+src[kyw+kxp1]+src[bw*kym1+kx]+src[bw*kyp1+kx]));
#endif

		kym1*=bw, kyp1*=bw;
		dst[kyw+kx]=src[kyw+kx] + filt[dim[1]]*(src[kyw+kxm1]+src[kyw+kxp1]+src[kym1+kx]+src[kyp1+kx]);
		//dst[kyw+kx]+=1;
	}
	kx+=!odd-odd;
#ifdef DEBUG_DWT
	printf("B: %d (%d, %d): %f\n", idx, kx, ky, src[kyw+kx]);
#endif
	if(kx<xsize)
		dst[kyw+kx]=src[kyw+kx];
		//dst[kyw+kx]+=2;
	++ky, kyw+=bw;
	if(ky<ysize)
	{
		if(kx<xsize)
		{
			kxm1=kx-1, kxp1=kx+1;
			kym1=ky-1, kyp1=ky+1;

			//mirror padding
			mask=kxm1<0, kxm1^=-mask, kxm1+=mask,  kxp1-=2&-(kxp1>=xsize);
			mask=kym1<0, kym1^=-mask, kym1+=mask,  kyp1-=2&-(kyp1>=ysize);
#ifdef DEBUG_DWT
			printf("C: %d (%d, %d) (%d->%d, %d->%d) %f %f %f\n", idx, kx, ky, kxm1, kxp1, kym1, kyp1, src[kyw+kx], filt[dim[1]],
				src[kyw+kx] + filt[dim[1]]*(src[kyw+kxm1]+src[kyw+kxp1]+src[bw*kym1+kx]+src[bw*kyp1+kx]));
#endif
		
			kym1*=bw, kyp1*=bw;
			dst[kyw+kx]=src[kyw+kx] + filt[dim[1]]*(src[kyw+kxm1]+src[kyw+kxp1]+src[kym1+kx]+src[kyp1+kx]);
			//dst[kyw+kx]+=3;
		}
		kx+=odd-!odd;
#ifdef DEBUG_DWT
		printf("D: %d (%d, %d) %f\n", idx, kx, ky, src[kyw+kx]);
#endif
		if(kx<xsize)
			dst[kyw+kx]=src[kyw+kx];
			//dst[kyw+kx]+=4;
	}
}//*/
/*__kernel void lift_2D(__global float *src, __global float *dst, __global int *dim, __global float *filt)
{
	int bw=dim[0], xsize=dim[2], ysize=dim[3], nxeven=dim[4], nyeven=dim[5];
	int idx=get_global_id(0);
	int kx=idx%xsize, ky=idx/xsize;
	int kxm1=kx-1, kxp1=kx+1,
		kym1=ky-1, kyp1=ky+1, mask,
		xodd=kx&1, yodd=ky&1;

	//mirror padding
	mask=kxm1<0, kxm1^=-mask, kxm1+=mask,  kxp1-=2&-(kxp1>=xsize);
	mask=kym1<0, kym1^=-mask, kym1+=mask,  kyp1-=2&-(kyp1>=ysize);

	kym1*=bw, ky*=bw, kyp1*=bw;
	dst[idx]=src[idx] + filt[xodd^yodd]*(src[ky+kxm1]+src[ky+kxp1]+src[kym1+kx]+src[kyp1+kx]);
}//*/
__kernel void permute_even_odd(__global float *src, __global float *dst, __global int *dim, __global float *gain)
{
	int bw=dim[DIM_BW], xsize=dim[DIM_XSIZE], nxeven=dim[DIM_XSIZE]-dim[DIM_NXODD], nyeven=dim[DIM_YSIZE]-dim[DIM_NYODD];
	int idx=get_global_id(0);
	int kx=idx%xsize, ky=idx/xsize;
	int xodd=kx&1, yodd=ky&1,
		kxd=(kx>>1)+(nxeven&-xodd), kyd=(ky>>1)+(nyeven&-yodd);

	dst[bw*kyd+kxd]=gain[xodd^yodd]*src[bw*ky+kx];
//	printf("%d, %d <- %d, %d: %f * %f\n", kxd, kyd, kx, ky, gain[0], src[bw*ky+kx]);
}
__kernel void permute_even_odd_inv(__global float *src, __global float *dst, __global int *dim, __global float *gain)
{
	int bw=dim[DIM_BW], xsize=dim[DIM_XSIZE], nxeven=dim[DIM_XSIZE]-dim[DIM_NXODD], nyeven=dim[DIM_YSIZE]-dim[DIM_NYODD];
	int idx=get_global_id(0);
	int kx=idx%xsize, ky=idx/xsize;
	int xodd=kx>=nxeven, yodd=ky>=nyeven,
		kxd=(kx-(nxeven&-xodd))<<1|xodd, kyd=(ky-(nyeven&-yodd))<<1|yodd;

	dst[bw*kyd+kxd]=gain[xodd^yodd]*src[bw*ky+kx];
//	printf("%d, %d <- %d, %d: %f * %f\n", kxd, kyd, kx, ky, gain[1], src[bw*ky+kx]);
}
__kernel void quantize(__global float *src, __global int *dst, __global int *range, __global float *gain)
{
	int idx=get_global_id(0);
	int result=(int)(gain[FILT_Q_AMP]*src[idx]), neg;
	result=clamp(result, range[0], range[1]);
	neg=result<0, result^=-neg, result+=neg, result=result<<1|neg;
	dst[idx]=result;
	//printf("%2d %10f -> %08X\n", idx, src[idx], dst[idx]);
}
__kernel void dequantize(__global int *src, __global float *dst, __global float *gain)
{
	int idx=get_global_id(0);
	int val=src[idx], neg=val&1;
	val>>=1, val^=-neg, val+=neg;
	dst[idx]=gain[FILT_Q_AMP]*val;
	//printf("%2d %10f <- %08X\n", idx, dst[idx], src[idx]);
}
__kernel void quantize_smooth(__global float *data, __global float *gain)
{
	int idx=get_global_id(0);
	float x=gain[FILT_Q_AMP]*data[idx], neg;
	x=0.07957747154594766788444188168626f*(x-sin(6.283185307179586476925286766559f*x));
	x=clamp(x, gain[FILT_Q_LO], gain[FILT_Q_HI]);
	data[idx]=x;
}
__kernel void dequantize_smooth(__global float *data, __global float *gain)
{
	int idx=get_global_id(0);
	//do not reverse the staircase, simply adjust the gain
	data[idx]*=gain[FILT_Q_AMP];
}
__kernel void sq_error(__global float *src, __global float *dst)
{
	int idx=get_global_id(0);
	float error=src[idx]-dst[idx];
	dst[idx]=error*error;
}

__kernel void conv33(__global float *src, __global float *dst, __global float *filt, __global int *dim)
{
	int idx=get_global_id(0);
	int kx=idx%dim[0], ky=idx/dim[0];
	int kxm1=kx-1, kxp1=kx+1, kym1=ky-1, kyp1=ky+1, mask;

	//mirror padding
	mask=kxm1<0, kxm1^=-mask, kxm1+=mask;
	kxp1-=2&-(kxp1>=dim[0]);

	mask=kym1<0, kym1^=-mask, kym1+=mask;
	kyp1-=2&-(kyp1>=dim[1]);

	kym1*=dim[0], ky*=dim[0], kyp1*=dim[0];
	dst[idx]=
		filt[0]*src[kym1+kxm1]+filt[1]*src[kym1+kx]+filt[2]*src[kym1+kxp1]+
		filt[3]*src[ky  +kxm1]+filt[4]*src[ky  +kx]+filt[5]*src[ky  +kxp1]+
		filt[6]*src[kyp1+kxm1]+filt[7]*src[kyp1+kx]+filt[8]*src[kyp1+kxp1];
}
#endif
