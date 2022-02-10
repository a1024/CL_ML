#include"main.h"
#include"OpenCL_wrap.h"
#include<stdio.h>
#include<stdarg.h>
#include<math.h>
#include<time.h>
#include<vector>
#include<iostream>
#include<map>
#include<algorithm>
#ifndef __ANDROID__
#include<tmmintrin.h>
#endif
#ifndef __linux__
#include<conio.h>
#include<Windows.h>
#endif
const char		file[]=__FILE__;

#ifdef _DEBUG
//	#define		DONT_LOAD_WEIGHTS
#endif

int				print_double(double x, int point_pos, int total)
{
//	int nbefore=sprintf_s(g_buf, G_BUF_SIZE, "%lld", (long long)abs(x));
	double a=abs(x);
	long long i=(long long)a;
	int nbefore=sprintf_s(g_buf, G_BUF_SIZE, "%lld", i);
	int nafter=sprintf_s(g_buf, G_BUF_SIZE, "%g", a-i)-2;
	int neg=x<0;
	int nspaces=point_pos-nbefore-neg;
	if(nspaces<0)
		nspaces=0;
	if(nspaces+neg+nafter>total)
		return printf("%*s%.*lf", nspaces, "", total-(nspaces+neg), x);
	int printed=printf("%*s%g", nspaces, "", x);
	if(printed<total)
		printed+=printf("%*s", total-printed, "");
	return printed;
}
void			print_data(const float *data, int iw, int ih, int x1, int x2, int y1, int y2, const char *format, ...)
{
	if(format)
	{
		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		printf("\n");
	}
	for(int ky=y1;ky<y2;++ky)
	{
		for(int kx=x1;kx<x2;++kx)
			print_double(data[iw*ky+kx], 4, 14);
		//	printf("%9g ", (double)data[iw*ky+kx]);
		printf("\n");
	}
	printf("\n");
}
#define			PRINT_CORNER(DATA, WIDTH, HEIGHT)		print_data(DATA, WIDTH, HEIGHT, 0, (int)minimum(WIDTH, 8), 0, (int)minimum(HEIGHT, 8), "Top-left corner preview out of %dx%d:", WIDTH, HEIGHT)
void			print_diff(const float *b1, const float *b2, int iw, int ih, int x1, int x2, int y1, int y2)
{
	for(int ky=y1;ky<y2;++ky)
	{
		for(int kx=x1;kx<x2;++kx)
			print_double(b1[iw*ky+kx]-b2[iw*ky+kx], 4, 12);
			//printf("%9g ", (double)(b1[iw*ky+kx]-b2[iw*ky+kx]));
		printf("\n");
	}
	printf("\n");
}
void			print_table(const float **table, int nrows, int ncolumns, const char *header)
{
	printf("%s", header);
	for(int kd=0;kd<nrows;++kd)
	{
		printf("%2d ", kd);
		for(int k2=0;k2<ncolumns;++k2)
			print_double(table[k2][kd], 4, 12);
		printf("\n");
	}
}

void			print_clmem_float(CLBuffer oclbuf, int iw, int ih, int x1, int x2, int y1, int y2, const char *name)
{
	auto buffer=oclbuf.read();
	print_data(buffer, iw, ih, x1, x2, y1, y2, name);
	delete[] buffer;
}
void			extract_channel(const int *buffer, float *data, int imsize, int channel)//[0, 255] -> [-1, 1[
{
	auto p=(const byte*)buffer+channel;
	const double inv128=1./128;
	for(int k=0;k<imsize;++k, p+=4)
		data[k]=float((*p-128)*inv128);
}
void			assign_channel(const float *data, int *buffer, int imsize, int channel)//[-1, 1[ -> [0, 255] clamped
{
	auto p=(byte*)buffer+channel;
	for(int k=0;k<imsize;++k, p+=4)
	{
		int val=int(data[k]*128)+128;
		if(val<0)
			val=0;
		if(val>255)
			val=255;
		*p=val;
	}
}
/*std::string		gen_filename()
{
	int s0=1024;
	std::string str(s0, '\0');

	time_t t=time(nullptr);
#ifdef __linux__
	auto &now=*localtime(&t);
#else
	tm now={};
	localtime_s(&now, &t);
#endif

	int printed=sprintf_s(&str[0], s0, "%04d%02d%02d_%02d%02d%02d_%g.PNG", 1900+now.tm_year, now.tm_mon+1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
}//*/
int*			save_data_rgb(const float *data, int dw, int dh, int nch, const char *filename=nullptr)
{
	MY_ASSERT(nch==3, "Can save data as image only with 3 channels\n");
	std::string str;
	if(filename)
		str=filename;
	else
		str=gen_filename();
	int size=dw*dh;
	auto buffer=new int[size];
	//memset(buffer, 0, size*sizeof(int));
	auto r=data, g=data+size, b=g+size;
	for(int k=0;k<size;++k)
		buffer[k]=0xFF000000|byte(128*b[k]+128)<<16|byte(128*g[k]+128)<<8|byte(128*r[k]+128);
	save_image(str.c_str(), buffer, dw, dh);
	return buffer;
}

WeightInfo		winfo[]=
{
#define			WNAME(NFILT, NCHAN, WIDTH, HEIGHT, PAD, STRIDE, FILENAME, LABEL)		{NFILT, NCHAN, WIDTH, HEIGHT, PAD, STRIDE, FILENAME},
#include		"names_resnet18.h"
#undef			WNAME
};
const char*		layer_name(WeightFileLabel label)
{
	const char *a="UNDEFINED";
	switch(label)
	{
#define			WNAME(NFILT, NCHAN, WIDTH, HEIGHT, PAD, STRIDE, FILENAME, LABEL)		case LABEL:a=#LABEL;break;
#include		"names_resnet18.h"
#undef			WNAME
	}
	return a;
}

typedef std::vector<float> vec;
typedef std::vector<int> veci;
void			lincoeff(vec const &w0, vec const &b0, vec const &mean0, vec const &var0, float *coeff, float *con)
{
	int size=(int)w0.size();
	for(int k=0;k<size;++k)
	{
		//y=w*(x-mean)/sqrt(var)+b
		float sdev=sqrt(var0[k]);
		coeff[k]=w0[k]/sdev;
		con[k]=b0[k]-mean0[k]/sdev;
	}
}
int				append_linop(vec const &w0, vec const &b0, vec const &mean0, vec const &var0, vec &weights, int &idx_m, int &idx_c)
{
	int s0=(int)weights.size(), delta=(int)w0.size();
	weights.resize(s0+(delta<<1));
	auto data=weights.data();
	lincoeff(w0, b0, mean0, var0, data+s0, data+s0+delta);
	idx_m=s0, idx_c=s0+delta;
	return delta;
}
int				load_bn(std::string const &wpath, WeightFileLabel *wbmv, vec &weights)
{
	vec w0, b0, mean0, var0;
	load_txt_gains(wpath, winfo[wbmv[0]], w0);
	load_txt_gains(wpath, winfo[wbmv[1]], b0);
	load_txt_gains(wpath, winfo[wbmv[2]], mean0);
	load_txt_gains(wpath, winfo[wbmv[3]], var0);
	int idx_m0=0, idx_c0=0;
	int delta=append_linop(w0, b0, mean0, var0, weights, idx_m0, idx_c0);
	return delta;
}

void			print_linop(float *coeff, float *con, int n)
{
	for(int k=0;k<n;++k)
		printf("%3d  %g x + %g\n", k, (double)coeff[k], (double)con[k]);
}


enum			LayerType
{
	L_INPUT,

	L_RES_SAVE_DS,
	L_RES_SAVE,
	L_RES_ADD,

	L_CONV,
	//L_BN,
	L_RELU,
	L_MAXPOOL,
	L_AVPOOL,
	L_LINEAR,
};
struct			Layer
{
	LayerType type;
	WeightFileLabel info[5];
	const char *filename;
	std::string to_string()
	{
		std::string str;
		switch(type)
		{
		case L_INPUT:		str+="L_INPUT";break;
		case L_RES_SAVE_DS:
		//	str+="L_RES_SAVE_DS";
			str+=filename;
			break;
		case L_RES_SAVE:	str+="L_RES_SAVE";break;
		case L_RES_ADD:		str+="L_RES_ADD";break;
		case L_CONV:
		//	str+="L_CONV";
			str+=filename;
			//std+=':';
			//str+=winfo[info[0]].filename;
			break;
		//case L_BN:
		//	str+="L_BN";
		//	//std+=':';
		//	//str+=winfo[info[0]].filename, str+="...";
		//
		//	//str+=winfo[info[0]].filename, str+=", ";
		//	//str+=winfo[info[1]].filename, str+=", ";
		//	//str+=winfo[info[2]].filename, str+=", ";
		//	//str+=winfo[info[3]].filename;
		//	break;
		case L_RELU:		str+="L_RELU";break;
		case L_MAXPOOL:		str+="L_MAXPOOL", str+=std::to_string((int)info[0]);break;
		case L_AVPOOL:		str+="L_AVPOOL", str+=std::to_string((int)info[0]);break;
		case L_LINEAR:		str+="L_LINEAR";
		}
		return str;
	}
};
enum	IndicesIdx//for indices array
{
	II_Cin, II_Win, II_Hin,
	II_Cout, II_Wout, II_Hout,
	II_logstride, II_Wk, II_Hk,
	II_bufsize,
};
struct			DataDim
{
	int nch;
	//short pad;
	int w, h;
	int total;
	DataDim():nch(0), w(0), h(0), total(0){}
	void calc_total()
	{
		total=nch*w*h;
		//total=nch*(w+2*pad)*(h+2*pad);
	}
	DataDim(short nch, int w, int h):nch(nch), w(w), h(h)
	{
		total=nch*w*h;
		//calc_total();
	}
	void set(short nch, int w, int h)
	{
		this->nch=nch;
		this->w=w, this->h=h;
		total=nch*w*h;
		//calc_total();
	}
	//size_t worksize()
	//{
	//	return nch*w*h;
	//	//return total;
	//}
	void set_in(int *indices)
	{
		indices[II_Cin]=nch;
		indices[II_Win]=w;
		indices[II_Hin]=h;
		//indices[II_Pin]=pad;
	}
	void set_out(int *indices)
	{
		indices[II_Cout]=nch;
		indices[II_Wout]=w;
		indices[II_Hout]=h;
		//indices[II_Pout]=pad;
	}
};
void			calc_datadim(Layer *net, int nlayers, DataDim *dim, int &maxidx)
{
	int maxvol=dim[0].total;
	maxidx=0;
	for(int k=0;k<nlayers;++k)
	{
		auto layer=net[k];
		auto &in=dim[k], &out=dim[k+1];
		switch(layer.type)
		{
		case L_INPUT:
		case L_RES_SAVE_DS:
		case L_RES_SAVE:
		case L_RES_ADD:
		//case L_BN:
		case L_RELU:
			out=in;
			break;
		case L_CONV:
			{
				auto &convinfo=winfo[layer.info[0]];
				MY_ASSERT(in.nch==convinfo.nchan, "%d != %d", in.nch, convinfo.nchan);
				//int nextpad=0;
				//{
				//	int k2=k+1;
				//	for(;k2<nlayers&&net[k2].type!=L_CONV;++k2);
				//	if(k2<nlayers)
				//		nextpad=winfo[net[k2].info[0]].pad;
				//}
				out.set(convinfo.nfilt, (in.w-convinfo.w+convinfo.pad*2)/convinfo.stride+1, (in.h-convinfo.h+convinfo.pad*2)/convinfo.stride+1);
				if(maxvol<out.total)
					maxvol=out.total, maxidx=k+1;
			}
			break;
		case L_MAXPOOL:
		case L_AVPOOL:
			{
				int poolsize=layer.info[0];
				MY_ASSERT(poolsize, "Pool size cannot be zero\n");
				out.set(in.nch, in.w/poolsize, in.h/poolsize);
			}
			break;
		case L_LINEAR:
			{
				auto &convinfo=winfo[layer.info[0]];
				MY_ASSERT(in.nch==convinfo.nchan, "%d != %d", in.nch, convinfo.nchan);
				out.set(convinfo.nfilt, 1, 1);
			}
			break;
		}
		printf("%3d  %s:\t%dx%dx%d\n", k, layer.to_string().c_str(), out.nch, out.w, out.h);
	}
	printf("max size at layer %d\n", maxidx);//
}
typedef std::vector<DataDim> vecd;
Layer			resnet18[]=
{
	{L_CONV, {CONV1_WEIGHT, BN1_WEIGHT, BN1_BIAS, BN1_RUN_MEAN, BN1_RUN_VAR}, "conv1.bin"},
	{L_RELU},
	{L_MAXPOOL, {(WeightFileLabel)2}},

	{L_RES_SAVE},
	{L_CONV, {L1_0_CONV1_WEIGHT, L1_0_BN1_WEIGHT, L1_0_BN1_BIAS, L1_0_BN1_RUN_MEAN, L1_0_BN1_RUN_VAR}, "L1.0.conv1.bin"},
	{L_RELU},
	{L_CONV, {L1_0_CONV2_WEIGHT, L1_0_BN2_WEIGHT, L1_0_BN2_BIAS, L1_0_BN2_RUN_MEAN, L1_0_BN2_RUN_VAR}, "L1.0.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},
	{L_RES_SAVE},
	{L_CONV, {L1_1_CONV1_WEIGHT, L1_1_BN1_WEIGHT, L1_1_BN1_BIAS, L1_1_BN1_RUN_MEAN, L1_1_BN1_RUN_VAR}, "L1.1.conv1.bin"},
	{L_RELU},
	{L_CONV, {L1_1_CONV2_WEIGHT, L1_1_BN2_WEIGHT, L1_1_BN2_BIAS, L1_1_BN2_RUN_MEAN, L1_1_BN2_RUN_VAR}, "L1.1.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},

	{L_RES_SAVE_DS, {L2_0_DS0_WEIGHT, L2_0_DS1_WEIGHT, L2_0_DS1_BIAS, L2_0_DS1_RUN_MEAN, L2_0_DS1_RUN_VAR}, "L2.0.ds1.bin"},
	{L_CONV, {L2_0_CONV1_WEIGHT, L2_0_BN1_WEIGHT, L2_0_BN1_BIAS, L2_0_BN1_RUN_MEAN, L2_0_BN1_RUN_VAR}, "L2.0.conv1.bin"},
	{L_RELU},
	{L_CONV, {L2_0_CONV2_WEIGHT, L2_0_BN2_WEIGHT, L2_0_BN2_BIAS, L2_0_BN2_RUN_MEAN, L2_0_BN2_RUN_VAR}, "L2.0.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},
	{L_RES_SAVE},
	{L_CONV, {L2_1_CONV1_WEIGHT, L2_1_BN1_WEIGHT, L2_1_BN1_BIAS, L2_1_BN1_RUN_MEAN, L2_1_BN1_RUN_VAR}, "L2.1.conv1.bin"},
	{L_RELU},
	{L_CONV, {L2_1_CONV2_WEIGHT, L2_1_BN2_WEIGHT, L2_1_BN2_BIAS, L2_1_BN2_RUN_MEAN, L2_1_BN2_RUN_VAR}, "L2.1.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},

	{L_RES_SAVE_DS, {L3_0_DS0_WEIGHT, L3_0_DS1_WEIGHT, L3_0_DS1_BIAS, L3_0_DS1_RUN_MEAN, L3_0_DS1_RUN_VAR}, "L3.0.ds1.bin"},
	{L_CONV, {L3_0_CONV1_WEIGHT, L3_0_BN1_WEIGHT, L3_0_BN1_BIAS, L3_0_BN1_RUN_MEAN, L3_0_BN1_RUN_VAR}, "L3.0.conv1.bin"},
	{L_RELU},
	{L_CONV, {L3_0_CONV2_WEIGHT, L3_0_BN2_WEIGHT, L3_0_BN2_BIAS, L3_0_BN2_RUN_MEAN, L3_0_BN2_RUN_VAR}, "L3.0.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},
	{L_RES_SAVE},
	{L_CONV, {L3_1_CONV1_WEIGHT, L3_1_BN1_WEIGHT, L3_1_BN1_BIAS, L3_1_BN1_RUN_MEAN, L3_1_BN1_RUN_VAR}, "L3.1.conv1.bin"},
	{L_RELU},
	{L_CONV, {L3_1_CONV2_WEIGHT, L3_1_BN2_WEIGHT, L3_1_BN2_BIAS, L3_1_BN2_RUN_MEAN, L3_1_BN2_RUN_VAR}, "L3.1.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},

	{L_RES_SAVE_DS, {L4_0_DS0_WEIGHT, L4_0_DS1_WEIGHT, L4_0_DS1_BIAS, L4_0_DS1_RUN_MEAN, L4_0_DS1_RUN_VAR}, "L4.0.ds1.bin"},
	{L_CONV, {L4_0_CONV1_WEIGHT, L4_0_BN1_WEIGHT, L4_0_BN1_BIAS, L4_0_BN1_RUN_MEAN, L4_0_BN1_RUN_VAR}, "L4.0.conv1.bin"},
	{L_RELU},
	{L_CONV, {L4_0_CONV2_WEIGHT, L4_0_BN2_WEIGHT, L4_0_BN2_BIAS, L4_0_BN2_RUN_MEAN, L4_0_BN2_RUN_VAR}, "L4.0.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},
	{L_RES_SAVE},
	{L_CONV, {L4_1_CONV1_WEIGHT, L4_1_BN1_WEIGHT, L4_1_BN1_BIAS, L4_1_BN1_RUN_MEAN, L4_1_BN1_RUN_VAR}, "L4.1.conv1.bin"},
	{L_RELU},
	{L_CONV, {L4_1_CONV2_WEIGHT, L4_1_BN2_WEIGHT, L4_1_BN2_BIAS, L4_1_BN2_RUN_MEAN, L4_1_BN2_RUN_VAR}, "L4.1.conv2.bin"},
	{L_RES_ADD},
	{L_RELU},

	{L_AVPOOL, {(WeightFileLabel)7}},

	{L_LINEAR, {FC_WEIGHT, FC_BIAS}, "fc.bin"},
};
const int		nlayers=SIZEOF(resnet18);
void			scale_nearest(const int *src, int sw, int sh, float *dst, int dw, int dh, const float *mean, const float *stdev)
{
	float gain=1.f/255;
	int outsize=dw*dh;
	auto r=dst, g=r+outsize, b=g+outsize;
	for(int ky=0, idx=0;ky<dh;++ky)
	{
		for(int kx=0;kx<dw;++kx, ++idx)
		{
			auto p=(unsigned char*)(src+sw*(ky*sh/dh)+kx*sw/dw);
			r[idx]=(p[0]*gain-mean[0])/stdev[0];
			g[idx]=(p[1]*gain-mean[1])/stdev[1];
			b[idx]=(p[2]*gain-mean[2])/stdev[2];
		}
	}
}
void			make_input(const int *buffer, int bw, int bh, int x0, int y0, int dx, int dy, float *dst)
{
	memset(dst, 0, dx*dy*sizeof(float));
	int xf1, xf2, yf1, yf2,
		xi1, xi2, yi1, yi2;
	if(x0<0)
		xi1=0, xf1=-x0;
	else
		xi1=x0, xf1=0;
	if(x0+dx>=bw)
		xi2=bw, xf2=bw-x0;
	else
		xi2=x0+dx, xf2=dx;
	
	if(y0<0)
		yi1=0, yf1=-y0;
	else
		yi1=y0, yf1=0;
	if(y0+dy>=bh)
		yi2=bh, yf2=bh-y0;
	else
		yi2=y0+dy, yf2=dy;

	float gain=1.f/128;
	for(int ys=yi1, yd=yf1;ys<yi2;++ys, ++yd)
	{
		for(int xs=xi1, xd=xf1;xs<xi2;++xs, ++xd)
		{
			auto p=(unsigned char*)(buffer+bw*ys+xs);
			dst[dx*(     yd)+xd]=(p[0]-128)*gain;
			dst[dx*(dy  +yd)+xd]=(p[1]-128)*gain;
			dst[dx*(dy*2+yd)+xd]=(p[2]-128)*gain;
		}
	}
}

typedef std::vector<size_t> vecs;
//void			set_idxbuf(CLBuffer ci, int *indices, DataDim const &in, DataDim const &out, int wk, int hk, int logstride, int debug_flag=false)
void			set_idxbuf(CLBuffer ci, int *indices, DataDim const &in, DataDim const &out, int wk, int hk, int logstride)
{
	indices[II_Cin]=in.nch;
	indices[II_Win]=in.w;
	indices[II_Hin]=in.h;
	//indices[II_Pin]=in.pad;
	indices[II_Cout]=out.nch;
	indices[II_Wout]=out.w;
	indices[II_Hout]=out.h;
	//indices[II_Pout]=out.pad;
	//indices[II_bias]=idx_bias;
	//indices[II_filt]=idx_filt;
	indices[II_logstride]=logstride;
	indices[II_Wk]=wk;
	indices[II_Hk]=hk;

	//if(debug_flag)//
	//	indices[II_Cout]=0;//

	ci.write(indices);

	//for(int k=0;k<II_bufsize;++k)
	//	printf("%d ", indices[k]);
	//printf("\n");
}
void			print_idxbuf(int *indices)
{
	printf("\n");
#define		PRINT(LABEL)	printf("%s = %d\n", #LABEL, indices[LABEL])
	PRINT(II_Cin);
	PRINT(II_Win);
	PRINT(II_Hin);
	PRINT(II_Cout);
	PRINT(II_Wout);
	PRINT(II_Hout);
	PRINT(II_logstride);
	PRINT(II_Wk);
	PRINT(II_Hk);
	PRINT(II_bufsize);
#undef		PRINT
	printf("\n");
}
void			calc_mean_var(float *data, int size, float &mean, float &stddev)
{
	mean=0;
	for(int k=0;k<size;++k)
		mean+=data[k];
	mean/=size;

	stddev=0;
	for(int k=0;k<size;++k)
	{
		float val=data[k]-mean;
		stddev+=val*val;
	}
	stddev=sqrt(stddev/size);
}

void			convert_all_weights_txt2bin(std::string const &wpath)
{
	vec convweights, bnweight, bnbias, bnmean, bnvar;
	vec resultweights;
	//vecs allidx;
	for(int kl=0;kl<nlayers;++kl)
	{
		convweights.clear();
		bnweight.clear();
		bnbias.clear();
		bnmean.clear();
		bnvar.clear();
		resultweights.clear();

		auto &layer=resnet18[kl];
		//printf("LAYER %d:\n", kl);
		if(layer.type==L_CONV||layer.type==L_RES_SAVE_DS)
		{
			auto &info=winfo[layer.info[0]];
			load_txt_weights_conv(wpath, info, convweights);
			load_txt_gains(wpath, winfo[layer.info[1]], bnweight);
			load_txt_gains(wpath, winfo[layer.info[2]], bnbias);
			load_txt_gains(wpath, winfo[layer.info[3]], bnmean);
			load_txt_gains(wpath, winfo[layer.info[4]], bnvar);
			int chsize=info.nchan*info.w*info.h;
			resultweights.reserve(info.nfilt*(chsize+1));
			for(int kc=0;kc<info.nfilt;++kc)
			{
				float w=bnweight[kc], b=bnbias[kc], m=bnmean[kc], v=bnvar[kc];
				float invstdev=1/sqrt(v+1e-7f);
				float gain=w*invstdev, bias=b-w*m*invstdev;//BN as conv gain & bias
				auto kernel=convweights.data()+chsize*kc;
				for(int k=0;k<chsize;++k)
					kernel[k]*=gain;
				resultweights.insert(resultweights.end(), kernel, kernel+chsize);
				resultweights.push_back(bias);
			}
#if 0
			for(int kc=0;kc<info.nfilt;++kc)
			{
				float mean=0, stddev=0;
				float *filter=resultweights.data()+(chsize+1)*kc;
				calc_mean_var(filter, chsize, mean, stddev);
				printf("%f\t%f\n", mean, stddev);
			}
			prompt("Continue?");
#endif

			//printf("First %dx%dx%d+1=%d filter CPU:\n", info.nchan, info.w, info.h, chsize+1);
			//for(int k=0;k<chsize+1;++k)//
			//	printf("%f\n", resultweights[k]);//
			//prompt("Continue?");

			auto success=save_weights_bin(resultweights.data(), (int)resultweights.size(), layer.filename, winfo[layer.info[0]], BINFILE_CONV);
			MY_ASSERT(success, "Couldn't save %s", layer.filename);
		}
		else if(layer.type==L_LINEAR)
		{
			auto &info=winfo[layer.info[0]];
			load_txt_weights_fc(wpath, info, resultweights);
			load_txt_gains(wpath, winfo[layer.info[1]], resultweights);
			auto success=save_weights_bin(resultweights.data(), (int)resultweights.size(), layer.filename, winfo[layer.info[0]], BINFILE_LINEAR);
			MY_ASSERT(success, "Couldn't save %s", layer.filename);
		}
	}
}
struct			SizeFactorization
{
	int size;
	short w, h;
};
SizeFactorization factorize_size(int size)
{
	SizeFactorization factorizations[]=
	{
		{3, 3, 1},
		{64, 8, 8},
		{128, 16, 8},
		{256, 16, 16},
		{512, 32, 16},
		{1000, 40, 25},
	};
	auto it=std::find_if(factorizations, factorizations+SIZEOF(factorizations), [&](SizeFactorization const &f)
	{
		return f.size==size;
	});
	return *it;
}
void			save_all_weights_png(std::vector<vec> &weights)
{
	if(weights.size()!=nlayers)
		return;
	for(int kl=0;kl<nlayers;++kl)
	{
		auto &wk=weights[kl];
		if(wk.size())
		{
			auto &layer=resnet18[kl];
			auto &info=winfo[layer.info[0]];
			if(layer.type==L_CONV||layer.type==L_RES_SAVE_DS)
			{
				int Cout=info.nfilt, Cin=info.nchan, W=info.w, H=info.h;
				auto outdim=factorize_size(Cout), indim=factorize_size(Cin);
				int kernelsize=W*H, filtsize=Cin*kernelsize, weightsize=Cout*filtsize;
				auto fimage=new unsigned char[weightsize], bimage=new unsigned char[Cout];
				int fimw=outdim.w*indim.w*W, fimh=outdim.h*indim.h*H;

				float fmin=wk[0], fmax=wk[0], bmin=wk[filtsize], bmax=wk[filtsize];
				for(int kco=0;kco<Cout;++kco)
				{
					auto filt=wk.data()+(filtsize+1)*kco;
					for(int k=0;k<filtsize;++k)
					{
						if(fmin>filt[k])
							fmin=filt[k];
						if(fmax<filt[k])
							fmax=filt[k];
					}
					if(bmin>filt[filtsize])
						bmin=filt[filtsize];
					if(bmax<filt[filtsize])
						bmax=filt[filtsize];
				}
				float fgain=fmin<fmax?255/(fmax-fmin):0;
				float bgain=bmin<bmax?255/(bmax-bmin):0;

				for(int kcoy=0;kcoy<outdim.h;++kcoy)
				{
					for(int kcox=0;kcox<outdim.w;++kcox)
					{
						int kco=outdim.w*kcoy+kcox;
						auto filt=wk.data()+(filtsize+1)*kco;
						for(int kciy=0;kciy<indim.h;++kciy)
						{
							for(int kcix=0;kcix<indim.w;++kcix)
							{
								int kci=indim.w*kciy+kcix;
								auto kernel=filt+kernelsize*kci;
								auto dst=fimage + fimw*( H*(indim.h*kcoy+kciy) ) + W*(indim.w*kcox+kcix);
								for(int ky=0;ky<H;++ky)
									for(int kx=0;kx<W;++kx)
										dst[fimw*ky+kx]=(unsigned char)(fgain*(kernel[W*ky+kx]-fmin));
							}
						}
						bimage[kco]=(unsigned char)(bgain*(filt[filtsize]-bmin));
					}
				}
				std::string fname=layer.filename, bname=layer.filename;
				fname+="_filt.PNG";
				bname+="_bias.PNG";
				save_image_monochrome(fname.c_str(), fimage, fimw, fimh);
				save_image_monochrome(bname.c_str(), bimage, outdim.w, outdim.h);
				delete[] fimage;
				delete[] bimage;
			}
			else if(layer.type==L_LINEAR)
			{
			}
		}
	}
}
size_t			load_all_weights_bin(std::string const &wpath, std::vector<vec> &weights)
{
	size_t nparams=0;
	weights.resize(nlayers);
	for(int k=0;k<nlayers;++k)
	{
		printf("\rLoading weights %d / %d [%.2lf%%]...", k+1, nlayers, 100.*(k+1)/nlayers);
		auto &wk=weights[k];
		auto &layer=resnet18[k];
		if(layer.type==L_CONV||layer.type==L_RES_SAVE_DS||layer.type==L_LINEAR)
		{
			load_weights_bin(wpath, layer.filename, winfo[layer.info[0]], wk);
#if 0
			auto &info=winfo[resnet18[k].info[0]];
			float sum=0, vmin=wk[0], vmax=wk[0];
			for(int k2=0;k2<(int)wk.size();++k2)
			{
				sum+=wk[k2];
				if(vmin>wk[k2])
					vmin=wk[k2];
				if(vmax<wk[k2])
					vmax=wk[k2];
			}
			printf("%3d: %d x %dx%dx%d: sum=%f min=%f, max=%f\n", k, info.nfilt, info.nchan, info.w, info.h, sum, vmin, vmax);
#endif
		}
		nparams+=wk.size();
	}
	printf("\n");
	return nparams;
}
void			send_all_weights_to_GPU(std::vector<vec> &weights, std::vector<CLBuffer> &cw)
{
	cw.resize(weights.size());
	for(int k=0;k<(int)weights.size();++k)
	{
		auto &wk=weights[k];
		if(wk.size())
		{
			//printf("%d: %d\n", k, (int)wk.size());
			cw[k].create(wk.size(), BUFFER_READ_WRITE);
			cw[k].write(wk.data());
		}
	}
}

void			print_GPU_usage()
{
	auto usage=ocl_query_mem_usage();
	int baseK=0;
	double units=(double)usage;
	for(;units>1024&&baseK<4;)
		units/=1024, ++baseK;
	const char *a=nullptr;
	switch(baseK)
	{
	case 0:a="bytes";break;
	case 1:a="KB";break;
	case 2:a="MB";break;
	case 3:a="GB";break;
	}
	printf("Using %.2lf %s of GPU memory\n", units, a);
}
void			print_mean_var(CLBuffer buf, int nc, int w, int h)
{
	auto csize=w*h, size=nc*csize;
	auto data=buf.read_sub(0, size);
	printf("Mean\tStd.dev\n");
	for(int kc=0;kc<nc;++kc)
	{
		float mean=0, stddev=0;
		calc_mean_var(data+csize*kc, csize, mean, stddev);
		printf("%f\t%f\n", mean, stddev);
	}
	prompt("Continue?");
}
void			print_GPU_buffer(CLBuffer buf, int nc, int w, int h)
{
	auto size=nc*w*h;
	auto temp=buf.read_sub(0, size);
	printf("\n");

	float vmin=temp[0], vmax=temp[0];
	for(int k=0;k<size;++k)
	{
		if(vmin>temp[k])
			vmin=temp[k];
		if(vmax<temp[k])
			vmax=temp[k];
	}
	printf("min=%f, max=%f\n", vmin, vmax);

	PRINT_CORNER(temp, w, h);
	delete[] temp;
}
void			save_GPU_buffer(CLBuffer clbuf, int nch, int w, int h, const char *layer_name)
{
	static int call_count=1;
	int cw=0, ch=0;//channel width & height
	cw=floor_log2(nch);
	if(nch==1<<cw)
	{
		ch=cw>>1;
		cw-=ch;
		cw=1<<cw, ch=1<<ch;
	}
	else if(nch==1000)
		cw=40, ch=25;
	else
	{
		do
		{
			printf("How to factorize %d?\n", nch);
			printf("Width: ");
			scanf_s("%d", &cw);
			printf("Height: ");
			scanf_s("%d", &ch);
		}while(cw*ch!=nch);
	}
	//printf("x = floor_log2(%d) = %d\nch>>1 = %d\ncw = %d, cw=%d, ch=%d\n", nch, floor_log2(nch), floor_log2(nch)>>1, floor_log2(nch)-(floor_log2(nch)>>1), cw, ch);//
	printf("\tSaving %dx%dx%d...\n", nch, w, h);
	//printf("Saving %dx%d data (%d channels)...\n", cw*w, ch*h, nch);
	int size=nch*w*h;
	auto data=clbuf.read_sub(0, size);
	auto buffer=new unsigned char[size];
	float vmin=data[0], vmax=data[0];
	for(int k=1;k<size;++k)
	{
		if(vmin>data[k])
			vmin=data[k];
		if(vmax<data[k])
			vmax=data[k];
	}
	if(vmax==vmin)
		memset(buffer, 0, size);
	else
	{
		float gain=255/(vmax-vmin);
		for(int kcy=0;kcy<ch;++kcy)
		{
			for(int kcx=0;kcx<cw;++kcx)
			{
				int kc=cw*kcy+kcx;
				for(int ky=0;ky<h;++ky)
				{
					for(int kx=0;kx<w;++kx)
					{
						float val=data[w*(h*kc+ky)+kx];
						buffer[w*(cw*(h*kcy+ky)+kcx)+kx]=(unsigned char)(gain*(val-vmin));
					}
				}
			}
		}
	}

	sprintf_s(g_buf, G_BUF_SIZE, "%03d_%s.PNG", call_count, layer_name);
	std::string filename=g_buf;
//	std::string filename=gen_filename(call_count);

	save_image_monochrome(filename.c_str(), buffer, cw*w, ch*h, false);
	delete[] data;
	delete[] buffer;
	++call_count;
}
int				softargmax(float *data, int size)
{
	int max_idx=0;
	float max=data[0];
	for(int k=1;k<size;++k)
		if(max<data[k])
			max=data[k], max_idx=k;
	float invmax=1/max;
	float sum=0;
	for(int k=0;k<size;++k)
	{
		data[k]=exp(invmax*data[k]);
		sum+=data[k];
	}
	float invsum=1/sum;
	for(int k=0;k<size;++k)
		data[k]*=invsum;
	
	max_idx=0, max=data[0];
	for(int k=1;k<size;++k)
		if(max<data[k])
			max=data[k], max_idx=k;

	return max_idx;
}
struct			ResultCandidate
{
	const char *name;
	int idx;
	float match;
};

void			conv_test()
{
	float srcfilt[]=
	{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10,
	};
	const int bw=4, bh=4,
		filtsize=3*3,
		datasize=bw*bh;
	float srcdata[]=
	{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	};
	int srcindices[]=
	{
		1, 4, 4,//in dim
		1, 4, 4,//out dim
		0,		//logstride
		3, 3,	//kernel size
	};

	ocl_init("cl_kernels.h");
	CLBuffer cfilt, cdata[2], cindices;
	cfilt.create(filtsize+1, BUFFER_READ_WRITE, srcfilt);
	cdata[0].create(datasize, BUFFER_READ_WRITE, srcdata);
	cdata[1].create(datasize, BUFFER_READ_WRITE);
	cindices.create(II_bufsize, BUFFER_READ_WRITE, srcindices);

	CLBuffer argbuf[]={cdata[0], cdata[1], cindices, cfilt};
	ocl_sync();
	kernels[OCL_conv_zp].call(datasize, argbuf, 4);

	auto result=cdata[1].read();
	print_data(result, bw, bh, 0, bw, 0, bh, "Conv result:");

	exit_success();
}
int				main(int argc, char **argv)
{
	set_console_buffer_size(120, 2000);
	printf("ML Sandbox\n\n");

	//conv_test();//

	const char kernels_srcname[]="cl_kernels.h";
	int success=file_is_readable(kernels_srcname)==1;
	MY_ASSERT(success, "\'%s\' not found\n", kernels_srcname);
	std::string input, wpath;
#ifdef __ANDROID__
	MY_ASSERT(argc>=2,
		"Usage:\n"
		"  ml64 weights [path/to/weights]\n"
		"  ml64 srcimage [path/to/weights]\n"
	);
#else
	if(argc<2)
	{
		printf("Input file: ");
		std::getline(std::cin, input);
		printf("Weights: ");
		std::getline(std::cin, wpath);
	}
#endif

#ifdef __ANDROID__
	wpath="/data/data/com.termux/files/home/cpp/resnet18/";
//#elif defined __linux__
//	MY_ASSERT(argc==2, "Please pass the path to the weights\n");
//	std::string wpath=argv[1];
#else
	if(argc==3)//if weights path not provided, assume weights are with the executable
		wpath=argv[2];
	else if(!wpath.size())
	{
		wpath=argv[0];
		int k=(int)wpath.size()-1;
		for(;k>=0&&wpath[k]!='/'&&wpath[k]!='\\';--k);
		++k;
		wpath.erase(k, wpath.size()-k);
		path_adjust(wpath);
	}
#endif

	path_adjust(wpath);
	printf("Weights path: %s\n", wpath.c_str());

#if 1
	if(argc>1&&!strcmp(argv[1], "weights"))
	{
		//wpath="E:/ML/resnet18/";//
		prompt("About to convert weights from text files to binary...");
		convert_all_weights_txt2bin(wpath);
		exit_success();
	}
#endif
	printf("Save data betweeen layers? [Y/N] ");
	char c=0;
	scanf_s("%c", &c, 1);
	bool save_internals=(c&0xDF)=='Y';

#ifndef DONT_LOAD_WEIGHTS
	std::vector<vec> weights;
	auto nparams=load_all_weights_bin(wpath, weights);
	printf("%d params\n", (int)nparams);
	//{
	//	save_all_weights_png(weights);//
	//	exit_success();//
	//}
#endif

	int maxidx=0;
	vecd datadim(nlayers+1);
	datadim[0].set(3, 224, 224);
	calc_datadim(resnet18, nlayers, datadim.data(), maxidx);
	int maxtotal=datadim[maxidx].total;


	int *buffer=nullptr, iw=0, ih=0;
	if(!input.size())
		input=argv[1];
	load_image(input.c_str(), buffer, iw, ih);
	int tch=datadim[0].nch, tw=datadim[0].w, th=datadim[0].h, ttotal=tch*tw*th;
	auto src=new float[ttotal];
	float stdev[]={0.229f, 0.224f, 0.225f};
	float mean[]={0.485f, 0.456f, 0.406f};
	scale_nearest(buffer, iw, ih, src, tw, th, mean, stdev);
	//make_input(buffer, iw, ih, 48, 159, tw, th, src);//kodim23.png
	PRINT_CORNER(src, tw, th);
	
	
	ocl_init(kernels_srcname);

	std::vector<CLBuffer> cweights;
	CLBuffer ci, ct1, ct2, ct3;
	ci.create(II_bufsize, BUFFER_READ_ONLY);
	ct1.create(maxtotal, BUFFER_READ_WRITE);
	ct2.create(maxtotal, BUFFER_READ_WRITE);
	ct3.create(maxtotal, BUFFER_READ_WRITE);
	
#ifndef DONT_LOAD_WEIGHTS
	send_all_weights_to_GPU(weights, cweights);
	//cw.create(weights.size(), BUFFER_READ_WRITE, weights.data());
	ocl_sync();
	//prompt("About to clear weights\n");
	weights.clear();
#endif

	print_GPU_usage();
	
	int indices[II_bufsize]={};
	CLBuffer argbuf[4]={};
	ct1.write_sub(src, 0, ttotal);
	indices[II_Win]=datadim[0].w;
	indices[II_Hin]=datadim[0].h;
	ci.write(indices);
	argbuf[0]=ct1;//src
	argbuf[1]=ct2;//dst
	argbuf[2]=ci;//indices
	
#ifndef DONT_LOAD_WEIGHTS
	auto t1=time_sec();
	for(int kl=0;kl<nlayers;++kl)
	{
		auto &layer=resnet18[kl];
		printf("\r%3d / %3d %s...", kl+1, nlayers, layer.to_string().c_str());
		//printf("args: %p, %p, %p, %p\n", argbuf[0], argbuf[1], argbuf[2], argbuf[3]);//
		auto &inshape=datadim[kl], &outshape=datadim[kl+1+(layer.type==L_RES_SAVE_DS)];
		auto &info=winfo[layer.info[0]];
		argbuf[3]=cweights[kl];//weights
		//print_GPU_buffer(ct1, inshape.nch, inshape.w, inshape.h);//
		//if(kl==36)//
		//	printf("%s: worksize=%d\nargbuf: %p %p %p %p\n", layer.to_string().c_str(), outshape.nch*outshape.w*outshape.h, argbuf[0], argbuf[1], argbuf[2], argbuf[3]);
		switch(layer.type)
		{
		case L_RES_SAVE_DS:
			{
				argbuf[0]=ct1;//src
				argbuf[1]=ct3;//dst
				//auto shape=outshape;
				//shape.w>>=1;
				//shape.h>>=1;
				//shape.nch<<=1;
				//shape.calc_total();//downsample dimensions
				set_idxbuf(ci, indices, inshape, outshape, 0, 0, floor_log2(info.stride));

				//print_idxbuf(indices);//

				ocl_sync();
				kernels[OCL_conv11].call(outshape.total, argbuf, 4);
				argbuf[0]=ct1;//src
				argbuf[1]=ct2;//dst
			}
			break;
		case L_RES_SAVE:
			ct3.copy_from(ct1);
			break;
		case L_RES_ADD:
			argbuf[0]=ct3;//src
			argbuf[1]=ct1;//dst
			set_idxbuf(ci, indices, inshape, outshape, 0, 0, 0);
			ocl_sync();
			kernels[OCL_res_add].call(outshape.nch*outshape.w*outshape.h, argbuf, 2);
			argbuf[0]=ct1;//src
			argbuf[1]=ct2;//dst
			break;
		case L_CONV:
			{
				//set_idxbuf(ci, indices, inshape, outshape, info.w, info.h, floor_log2(info.stride), kl==36);//
				set_idxbuf(ci, indices, inshape, outshape, info.w, info.h, floor_log2(info.stride));
				//for(int k=0;k<II_bufsize;++k)//
				//	printf("%d ", indices[k]);
				//printf("\n");//
				int k_idx=0;
				switch(info.w)
				{
				case 1:
					k_idx=OCL_conv11;
					break;
				default:
					k_idx=OCL_conv_zp;
					break;
				}
				ocl_sync();
				kernels[k_idx].call(outshape.nch*outshape.w*outshape.h, argbuf, 4);
				std::swap(ct1, ct2);
				argbuf[0]=ct1;//src
				argbuf[1]=ct2;//dst
				//prompt("Continue?\n");
			}
			break;
		case L_RELU:
			ocl_sync();
			kernels[OCL_relu].call(outshape.nch*outshape.w*outshape.h, argbuf, 1);
			break;
		case L_MAXPOOL:
			set_idxbuf(ci, indices, inshape, outshape, 2, 2, 0);
			ocl_sync();
			kernels[OCL_maxpool].call(outshape.nch*outshape.w*outshape.h, argbuf, 3);
			std::swap(ct1, ct2);
			argbuf[0]=ct1;//src
			argbuf[1]=ct2;//dst
			break;
		case L_AVPOOL:
			set_idxbuf(ci, indices, inshape, outshape, 7, 7, 0);
			ocl_sync();
			kernels[OCL_avpool].call(outshape.nch*outshape.w*outshape.h, argbuf, 3);
			std::swap(ct1, ct2);
			argbuf[0]=ct1;//src
			argbuf[1]=ct2;//dst
			break;
		case L_LINEAR:
			set_idxbuf(ci, indices, inshape, outshape, 7, 7, 0);
			ocl_sync();
			kernels[OCL_linear_fc].call(outshape.nch*outshape.w*outshape.h, argbuf, 4);
			std::swap(ct1, ct2);
			argbuf[0]=ct1;//src
			argbuf[1]=ct2;//dst
			break;
		}
#if 0
		//print_mean_var(ct1, outshape.nch, outshape.w, outshape.h);
		print_GPU_buffer(ct1, outshape.nch, outshape.w, outshape.h);//
#endif
	//	if(save_internals&&layer.type==L_RELU)//save the output of each relu
		if(save_internals)
		{
			CLBuffer buf=layer.type==L_RES_SAVE_DS||layer.type==L_RES_SAVE?ct3:ct1;
			save_GPU_buffer(buf, outshape.nch, outshape.w, outshape.h, layer.to_string().c_str());
		}
	}
	auto t2=time_sec();
	printf("\nResNet evaluation elapsed: %lf ms\n", 1000*(t2-t1));

	auto r_size=datadim.back().total;
	auto result=ct1.read_sub(0, r_size);
	int idx=softargmax(result, r_size);

	auto table=new ResultCandidate[r_size];
	for(int k=0;k<r_size;++k)
	{
		table[k].name=classtable[k].name;
		table[k].idx=k;
		table[k].match=result[k];
	}
	std::sort(table, table+r_size, [](ResultCandidate const &a, ResultCandidate const &b){return a.match>b.match;});//descending order
	printf("\nClass  Match  Description\n");
	for(int k=0;k<r_size;++k)
		printf("%3d  %f  %s\n", table[k].idx, table[k].match, table[k].name);

	//print_data(result, 1, r_size, 0, 1, 0, r_size, "Results:\n");
	//printf("Highest score: class %d: %s\n", idx, classtable[idx].name);
	delete[] result;
#endif
	
	for(int k=0;k<(int)cweights.size();++k)
		cweights[k].release();
	//cw.release();
	ct1.release();
	ct2.release();
	ct3.release();
	ci.release();
	ocl_finish();

	delete[] src;
	free(buffer);

	exit_success();
	return 0;
}
