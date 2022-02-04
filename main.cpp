#include"main.h"
#include"OpenCL_wrap.h"
#include<stdio.h>
#include<stdarg.h>
#include<math.h>
#include<time.h>
#include<vector>
#include<iostream>
#include<map>
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
	//void calc_total()
	//{
	//	total=nch*(w+2*pad)*(h+2*pad);
	//}
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
	size_t worksize()
	{
		return total;
		//return nch*w*h;
	}
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
#if 0
enum			LayerType
{
	L_INPUT,
	L_NORMAL,
	L_RESIDUAL,

	SL_CONV,
	SL_BN,
	SL_RELU,
	SL_MAXPOOL,
	SL_AVPOOL,
};
struct			Layer
{
	LayerType type;
	int nsublayers;
	LayerType sub[6];
	WeightFileLabel info[24];
};
struct			LayerIdx
{
	short layer, sublayer;
	LayerIdx():layer(0), sublayer(0){}
	LayerIdx(short layer, short sublayer):layer(layer), sublayer(sublayer){}
	void set(short layer, short sublayer)
	{
		this->layer=layer, this->sublayer=sublayer;
	}
};
struct			DataDim
{
	LayerIdx resultof;
	short nch, pad;
	int w, h, total;
	DataDim():nch(0), pad(0), w(0), h(0), total(0){}
	void calc_total()
	{
		total=nch*(w+2*pad)*(h+2*pad);
	}
	DataDim(short layer, short sublayer, short nch, short pad, int w, int h):resultof(layer, sublayer), nch(nch), pad(pad), w(w), h(h)
	{
		calc_total();
	}
	void set(short layer, short sublayer, short nch, short pad, int w, int h)
	{
		resultof.set(layer, sublayer);
		this->nch=nch, this->pad=pad;
		this->w=w, this->h=h;
		calc_total();
	}
};
typedef std::vector<DataDim> vecd;
/*void			calc_datadim(WeightFileLabel *net, int nlayers, DataDim *dim, int &maxidx)
{
	int maxvol=dim[0].total;
	maxidx=0;
	for(int k=0;k<nlayers;++k)
	{
		auto &layer=winfo[net[k]];
		auto &in=dim[k], &out=dim[k+1];
		MY_ASSERT(in.nch==layer.nchan, "%d != %d", in.nch, layer.nchan);
		int nextpad=k+1<nlayers?winfo[net[k+1]].pad:0;
		out.set(layer.nfilt, nextpad, (in.w-layer.w+layer.pad*2)/layer.stride+1, (in.h-layer.h+layer.pad*2)/layer.stride+1);
		if(maxvol<out.total)
			maxvol=out.total, maxidx=k+1;
		printf("%d \'%s\' %d x %d x %d, total %d\n", k, layer_name(net[k]), out.nch, out.w, out.h, out.total);//
	}
	printf("max size at layer %d\n", maxidx);//
}
WeightFileLabel resnet18_summary[]=
{
	CONV1_WEIGHT,
	MAXPOOL2,
	
	L1_0_CONV1_WEIGHT,
	L1_0_CONV2_WEIGHT,
	L1_1_CONV1_WEIGHT,
	L1_1_CONV2_WEIGHT,
	
	L2_0_CONV1_WEIGHT,
	L2_0_CONV2_WEIGHT,
	L2_1_CONV1_WEIGHT,
	L2_1_CONV2_WEIGHT,

	L3_0_CONV1_WEIGHT,
	L3_0_CONV2_WEIGHT,
	L3_1_CONV1_WEIGHT,
	L3_1_CONV2_WEIGHT,

	L4_0_CONV1_WEIGHT,
	L4_0_CONV2_WEIGHT,
	L4_1_CONV1_WEIGHT,
	L4_1_CONV2_WEIGHT,
};
int				convlayers=SIZEOF(resnet18_summary);//*/
void			calc_datadim(Layer *net, int nlayers, DataDim *dim, int &maxidx)
{
	//MY_ASSERT(dim.size()==1, "Data dimension vector should contain only the input dimensions\n");
	int maxvol=dim[0].total;
	maxidx=0;
	for(int kl=0;kl<nlayers;++kl)
	{
		auto &layer=net[kl];
		if(layer.type==L_RESIDUAL)
		{
			auto &in=dim[kl], &out=dim[kl+1];
			for(int ks=0;ks<layer.nsublayers;++ks)
			{
				switch(layer.sub[ks])
				{
				case SL_CONV:
					{
						auto &sub=winfo[layer.info[ks<<2]];
						MY_ASSERT(in.nch==sub.nchan, "%d != %d", in.nch, sub.nchan);
						int nextpad=0;
						out.set(kl, ks, sub.nfilt, nextpad, (in.w-sub.w+sub.pad*2)/sub.stride+1, (in.h-sub.h+sub.pad*2)/sub.stride+1);
					}
					break;
				case SL_MAXPOOL:
				case SL_AVPOOL:
					break;
				}
			}
		}
		//auto &layer=winfo[net[k]];
		//auto &in=dim[k], &out=dim[k+1];
		//MY_ASSERT(in.nch==layer.nchan, "%d != %d", in.nch, layer.nchan);
		//int nextpad=k+1<nlayers?winfo[net[k+1]].pad:0;
		//out.set(layer.nfilt, nextpad, (in.w-layer.w+layer.pad*2)/layer.stride+1, (in.h-layer.h+layer.pad*2)/layer.stride+1);
		//if(maxvol<out.total)
		//	maxvol=out.total, maxidx=k+1;
		//printf("%d \'%s\' %d x %d x %d, total %d\n", k, layer_name(net[k]), out.nch, out.w, out.h, out.total);//
	}
	//printf("max size at layer %d\n", maxidx);//
}
Layer			resnet18[]=
{
	{
		L_NORMAL, 4, {SL_CONV, SL_BN, SL_RELU, SL_MAXPOOL},
		{
			CONV1_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			BN1_WEIGHT, BN1_BIAS, BN1_RUN_MEAN, BN1_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,
			(WeightFileLabel)2, LB_NONE, LB_NONE, LB_NONE,
		}
	},
	{
		L_RESIDUAL, 6, {SL_CONV, SL_BN, SL_RELU, SL_CONV, SL_BN, SL_RELU},
		{
			L1_0_CONV1_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L1_0_BN1_WEIGHT, L1_0_BN1_BIAS, L1_0_BN1_RUN_MEAN, L1_0_BN1_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,

			L1_0_CONV2_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L1_0_BN2_WEIGHT, L1_0_BN2_BIAS, L1_0_BN2_RUN_MEAN, L1_0_BN2_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,
		}
	},
	{
		L_RESIDUAL, 6, {SL_CONV, SL_BN, SL_RELU, SL_CONV, SL_BN, SL_RELU},
		{
			L2_0_CONV1_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L2_0_BN1_WEIGHT, L2_0_BN1_BIAS, L2_0_BN1_RUN_MEAN, L2_0_BN1_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,

			L2_0_CONV2_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L2_0_BN2_WEIGHT, L2_0_BN2_BIAS, L2_0_BN2_RUN_MEAN, L2_0_BN2_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,
		}
	},
	{
		L_RESIDUAL, 6, {SL_CONV, SL_BN, SL_RELU, SL_CONV, SL_BN, SL_RELU},
		{
			L3_0_CONV1_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L3_0_BN1_WEIGHT, L3_0_BN1_BIAS, L3_0_BN1_RUN_MEAN, L3_0_BN1_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,

			L3_0_CONV2_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L3_0_BN2_WEIGHT, L3_0_BN2_BIAS, L3_0_BN2_RUN_MEAN, L3_0_BN2_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,
		}
	},
	{
		L_RESIDUAL, 6, {SL_CONV, SL_BN, SL_RELU, SL_CONV, SL_BN, SL_RELU},
		{
			L4_0_CONV1_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L4_0_BN1_WEIGHT, L4_0_BN1_BIAS, L4_0_BN1_RUN_MEAN, L4_0_BN1_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,

			L4_0_CONV2_WEIGHT, LB_NONE, LB_NONE, LB_NONE,
			L4_0_BN2_WEIGHT, L4_0_BN2_BIAS, L4_0_BN2_RUN_MEAN, L4_0_BN2_RUN_VAR,
			LB_NONE, LB_NONE, LB_NONE, LB_NONE,
		}
	},
	{
		L_NORMAL, 1, {SL_AVPOOL}, {(WeightFileLabel)7},
	},
};
#endif
void			scale_nearest(const int *src, int sw, int sh, float *dst, int dw, int dh)
{
	float gain=1.f/128;
	int outsize=dw*dh;
	auto r=dst, g=r+outsize, b=g+outsize;
	for(int ky=0, idx=0;ky<dh;++ky)
	{
		for(int kx=0;kx<dw;++kx, ++idx)
		{
			auto p=(unsigned char*)(src+sw*(ky*sh/dh)+kx*sw/dw);
			r[idx]=(p[0]-128)*gain;
			g[idx]=(p[1]-128)*gain;
			b[idx]=(p[2]-128)*gain;
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
void			set_idxbuf(CLBuffer ci, int *indices, DataDim const &in, DataDim const &out, int wk, int hk, int logstride, int debug_flag=false)
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

	if(debug_flag)//
		indices[II_Cout]=0;//

	ci.write(indices);

	//for(int k=0;k<II_bufsize;++k)
	//	printf("%d ", indices[k]);
	//printf("\n");
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
				resultweights.insert(resultweights.begin(), kernel, kernel+chsize);
				resultweights.push_back(bias);
			}
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
size_t			load_all_weights_bin(std::string const &wpath, std::vector<vec> &weights)
{
	size_t nparams=0;
	weights.resize(nlayers);
	for(int k=0;k<nlayers;++k)
	{
		auto &wk=weights[k];
		auto &layer=resnet18[k];
		if(layer.type==L_CONV||layer.type==L_RES_SAVE_DS||layer.type==L_LINEAR)
			load_weights_bin(wpath, layer.filename, winfo[layer.info[0]], wk);
		nparams+=wk.size();
	}
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
void			print_GPU_buffer(CLBuffer buf, int w, int h)
{
	auto temp=buf.read_sub(0, w*h);
	printf("\n");
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
	printf("Saving %dx%d data (%d channels)...\n", cw*w, ch*h, nch);
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
int				main(int argc, char **argv)
{
	set_console_buffer_size(120, 2000);
	printf("ML Sandbox\n\n");
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

#if 1
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
	if(!strcmp(argv[1], "weights"))
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
	//vec weights;
	//vecs allidx;
	auto nparams=load_all_weights_bin(wpath, weights);
	//load_all_weights_bin(wpath, weights, allidx);
	printf("%d params\n", (int)nparams);
#endif
/*	for(int k=0;k<nlayers;++k)
	{
		auto &layer=resnet18[k];
		printf("LAYER %d:\n", k);
		switch(layer.type)
		{
		case L_INPUT:break;
		
		case L_RES_SAVE_DS:break;
		case L_RES_SAVE:break;
		case L_RES_ADD:break;
		
		case L_CONV:
			allidx.push_back(weights.size());
			load_weights_txt(wpath, winfo[layer.info[0]], weights);
			break;
		case L_BN:
			load_bn(wpath, layer.info, weights);
			break;
		case L_RELU:break;
		case L_MAXPOOL:break;
		case L_AVPOOL:break;
		}
	}//*/
	
/*	vec weights, w0, b0, mean0, var0;
	load_gains(wpath, winfo[BN1_WEIGHT], w0);
	load_gains(wpath, winfo[BN1_BIAS], b0);
	load_gains(wpath, winfo[BN1_RUN_MEAN], mean0);
	load_gains(wpath, winfo[BN1_RUN_VAR], var0);
	int idx_m0=0, idx_c0=0;
	int size0=append_linop(w0, b0, mean0, var0, weights, idx_m0, idx_c0);
	//print_linop(weights.data()+idx_m0, weights.data()+idx_c0, size0);
*/
	//WeightFileLabel labelbn1[]={BN1_WEIGHT, BN1_BIAS, BN1_RUN_MEAN, BN1_RUN_VAR};
	//load_bn(wpath, labelbn1, weights);
	//load_weights(wpath, winfo[CONV1_WEIGHT], weights);

	int maxidx=0;
	vecd datadim(nlayers+1);
	datadim[0].set(3, 224, 224);
	calc_datadim(resnet18, nlayers, datadim.data(), maxidx);
	int maxtotal=datadim[maxidx].total;
	//int maxidx=0;
	//vecd datadim(convlayers+1);
	//datadim[0].set(3, 3, 224, 224);
	//calc_datadim(resnet18_summary, convlayers, datadim.data(), maxidx);
	//int maxtotal=datadim[maxidx].total;


	int *buffer=nullptr, iw=0, ih=0;
	load_image(argv[1], buffer, iw, ih);
	int tch=datadim[0].nch, tw=datadim[0].w, th=datadim[0].h, ttotal=tch*tw*th;
	auto src=new float[ttotal];
	scale_nearest(buffer, iw, ih, src, tw, th);
	//make_input(buffer, iw, ih, 48, 159, tw, th, src);
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
		//if(kl==37)//
		//	debug_flag=true;//
		printf("\r%3d / %3d %s...", kl+1, nlayers, layer.to_string().c_str());
		//printf("args: %p, %p, %p, %p\n", argbuf[0], argbuf[1], argbuf[2], argbuf[3]);//
		auto &inshape=datadim[kl], &outshape=datadim[kl+1];
		auto &info=winfo[layer.info[0]];
		argbuf[3]=cweights[kl];//weights
#if 0
		print_GPU_buffer(ct1, inshape.w, inshape.h);//
#endif
		//if(kl==36)//
		//	printf("%s: worksize=%d\nargbuf: %p %p %p %p\n", layer.to_string().c_str(), outshape.nch*outshape.w*outshape.h, argbuf[0], argbuf[1], argbuf[2], argbuf[3]);
		switch(layer.type)
		{
		case L_RES_SAVE_DS:
			argbuf[0]=ct1;//src
			argbuf[1]=ct3;//dst
			set_idxbuf(ci, indices, inshape, outshape, 0, 0, floor_log2(info.stride));
			ocl_sync();
			kernels[OCL_conv11].call(outshape.nch*(outshape.w>>1)*(outshape.h>>1), argbuf, 4);//downsample dimensions
			argbuf[0]=ct1;//src
			argbuf[1]=ct2;//dst
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
				set_idxbuf(ci, indices, inshape, outshape, info.w>>1, info.h>>1, floor_log2(info.stride), kl==36);//
			//	set_idxbuf(ci, indices, inshape, outshape, info.w>>1, info.h>>1, floor_log2(info.stride));
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
		print_GPU_buffer(ct1, outshape.w, outshape.h);//
#endif
		if(save_internals)//save the output of each relu
	//	if(save_internals&&layer.type==L_RELU)//save the output of each relu
			save_GPU_buffer(layer.type==L_RES_SAVE?ct3:ct1, outshape.nch, outshape.w, outshape.h, layer.to_string().c_str());
	}
	auto t2=time_sec();
	printf("ResNet evaluation elapsed: %lf ms\n", 1000*(t2-t1));

	auto r_size=datadim.back().total;
	auto result=ct1.read_sub(0, r_size);
	int idx=softargmax(result, r_size);
	print_data(result, 1, r_size, 0, 1, 0, r_size, "Results:\n");
	printf("Highest score: class %d: %s\n", idx, classtable[idx].name);
	delete[] result;
#endif
	
//	auto data0=ct2.read_sub(0, datadim[0].total);
//	int w0=datadim[0].w+datadim[0].pad*2, h0=datadim[0].h+datadim[0].pad*2, nch0=datadim[0].nch;
//	PRINT_CORNER(data0, w0, h0);
//	save_data_rgb(data0, w0, h0, nch0);
//	//print_data(data0, w0, h0, 0, w0, 0, h0, "Padded data");
//	exit_success();//

	//int ii=0;
	//set_idxbuf(indices, datadim[0], datadim[1], (int)allidx[ii], 0, 1);
	//ci.write(indices);
	//argbuf[0]=ct1;
	//argbuf[1]=ct2;
	//argbuf[2]=cw;
	//argbuf[3]=ci;
	//kernels[OCL_conv_n77_zp].call(datadim[1].worksize(), argbuf, 4);
	
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
	
	//int s0=weights.size();
	//weights.resize(s0+(w0.size()<<1));
	//lincoeff(w0, b0, mean0, var0, weights.data()+s0
	
	//std::vector<float> weight0, bias0, mean0, var0;
	//int size;
	//load_gains(weightspath, "layer1.0.bn1.weight.txt", weight0, size);
	//MY_ASSERT(size==64, "");
	//load_gains(weightspath, "layer1.0.bn1.bias.txt", bias0, size);
	//MY_ASSERT(size0==size1, "");

	//get_path(weightspath, "Weights folder: ");

	//int nfilters=0, nchannels=0, w=0, h=0;
	//std::vector<float> l1c1;
	//load_weights(weightspath, "layer1.0.conv1.weight.txt", l1c1, nfilters, nchannels, w, h);

	//std::string inpath, outpath;
	//get_path(inpath, "Input folder: ");
	//get_path(outpath, "Output folder: ");
	//ocl_init("cl_kernels.h");
	//ocl_finish();
#endif

#if 0
#if 1
#define IMAGE_LOADED
	int *buffer=nullptr, iw=0, ih=0;
	load_image(argv[1], buffer, iw, ih);
#else
	int iw=8, ih=8;
	int buffer[]=
	{
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255,
		//0, 0, 0, 0,   0, 0, 0, 0,
		//0, 0, 0, 0,   0, 0, 0, 0,
		//0, 0, 0, 0,   0, 0, 0, 0,
		//0, 0, 0, 0,   0, 0, 0, 0,
		//0, 0, 0, 0, 255, 0, 0, 0,
		//0, 0, 0, 0,   0, 0, 0, 0,
		//0, 0, 0, 0,   0, 0, 0, 0,
		//0, 0, 0, 0,   0, 0, 0, 0,
	};
#endif
	int imsize=iw*ih;

	//for(int k=0;k<imsize;++k)//
	//	buffer[k]=rand();
	//	//buffer[k]=k;//

	ocl_init();

	auto data=new float[imsize], d2=new float[imsize];
	extract_channel(buffer, data, imsize, 0);
	//for(int k=0;k<imsize;++k)
	//	data[k]=k/64.f;

	//learned parameters:	2 norms + (even) filt size
	float filt[]=
	{
		// 1.20382,		//ratio=16.9183, dist=0.119696
		// 1.08005,
		//-0.206265,
		//-0.266164,
		//-0.172423,
		//-0.270596,

		//1, 1, 0, 0, 0, 0,

		 1.1496043988602962433651033986614476f,
		 0.86986445162473959153241758552174375f,

		-1.5861343420594238292020515785937243f,
		-0.052980118573376671019344897514278511f,
		 0.882911075528503100647487289764588328f,
		 0.443506852044983007635941975182636061f,

		// 0.1,//leaky ReLU gain

		//-0.5f, 0.25f, -0.125f, 0.0625f,
		//127,//quantization amplitude
	};
	const int filtsize=SIZEOF(filt);

	printf("Allocating GPU buffers...\n");
	int error=0;
	oclim0		=p_clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, imsize*sizeof(float), data, &error);CL_CHECK(error);
	oclsrcbuf	=p_clCreateBuffer(context, CL_MEM_READ_WRITE, imsize*sizeof(float), nullptr, &error);	CL_CHECK(error);
	ocldstbuf	=p_clCreateBuffer(context, CL_MEM_READ_WRITE, imsize*sizeof(float), nullptr, &error);	CL_CHECK(error);
	ocldim		=p_clCreateBuffer(context, CL_MEM_READ_ONLY, 7*sizeof(int), nullptr, &error);			CL_CHECK(error);
	oclfilt		=p_clCreateBuffer(context, CL_MEM_READ_ONLY, filtsize*sizeof(float), nullptr, &error);	CL_CHECK(error);

	auto f2=new float[filtsize];
	memcpy(f2, filt, filtsize*sizeof(float));
	float beta[]={0.9f, 0.999f};
	//float grad[filtsize];
	//printf("Entering calc_loss...\n");
	auto t1=__rdtsc();
	training_adam(data, d2, buffer, iw, ih, f2, filtsize, beta, 0.001f, 0.0001f, 50);
	//calc_grad(data, d2, buffer, iw, ih, filt, SIZEOF(filt), grad);
	//auto L=calc_loss(data, d2, buffer, iw, ih, filt, SIZEOF(filt));
	auto t2=__rdtsc();

	//print_grad(filt, grad, filtsize);
	//print_data(grad, 1, filtsize, 0, 1, 0, filtsize, "Gradient:");
	//printf("Loss: %lf\n", L);
	printf("Elapsed: %lld\n", t2-t1);
	float *table[]={filt, f2};
	print_table((const float**)table, filtsize, SIZEOF(table), "dim\told\tnew\n");//

	receive_buffer(ocldstbuf, d2, imsize);
	for(int k=0;k<imsize;++k)
		buffer[k]=0xFF000000;
	//memset(buffer, 0, imsize*sizeof(int));
	assign_channel(d2, buffer, imsize, 0);

	gen_filename();
	save_image(g_buf, buffer, iw, ih);

#ifdef IMAGE_LOADED
	STBI_FREE(buffer);
#endif
	error=p_clReleaseMemObject(oclim0);		CL_CHECK(error);
	error=p_clReleaseMemObject(oclsrcbuf);	CL_CHECK(error);
	error=p_clReleaseMemObject(ocldstbuf);	CL_CHECK(error);
	error=p_clReleaseMemObject(ocldim);		CL_CHECK(error);
	error=p_clReleaseMemObject(oclfilt);	CL_CHECK(error);
	delete[] data, d2;
	ocl_finish();
#endif

#if 0
#if 1
	if(argc!=2)
		return 1;
	int iw=0, ih=0, nch=0;
	auto original_image=stbi_load(argv[1], &iw, &ih, &nch, 4);
	if(!original_image)
	{
		printf("Couldn't load image\n");
		return 1;
	}
	int imsize=iw*ih;
	auto buffer=(int*)original_image;
	auto src=new float[imsize], dst=new float[imsize];
	for(int k=0;k<imsize;++k)
		src[k]=(buffer[k]&0xFF)*(1.f/255);
	STBI_FREE(original_image);
#endif
#if 0
	if(argc!=2)
		return 1;
	std::vector<unsigned char> result;
	std::string filename=argv[1];
	lodepng::load_file(result, filename);
	unsigned iw, ih;
	int error=lodepng::decode(result, iw, ih, filename, LCT_RGBA, 8);//result contains the PNG header
	if(error)
	{
		printf("Couldn't load image\n");
		return 1;
	}
	int imsize=iw*ih;
	auto buffer=(int*)result.data();
	auto src=new float[imsize], dst=new float[imsize];
	for(int k=0;k<imsize;++k)
		src[k]=(buffer[k]&0xFF)*(1.f/255);
	memset(dst, 0, imsize*sizeof(float));
#endif

#if 0
	const int iw=12, ih=12;
	//const int iw=60, ih=32;
	short buffer[]=
	{
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,

		//firefox logo, w=60, h=32, depth=8bit
		//208, 208, 207, 207, 206, 205, 204, 204, 202, 201, 200, 198, 197, 195, 193, 190, 189, 184, 182, 179, 178, 175, 173, 170, 168, 167, 165, 160, 158, 157, 155, 153, 151, 150, 148, 146, 146, 144, 142, 141, 140, 140, 142, 171,  46, 128, 128, 129, 153,  56,  90,  90,  18, 219, 207, 204, 205, 206, 206, 206,
		//204, 203, 203, 202, 201, 200, 199, 199, 197, 195, 194, 193, 190, 189, 186, 185, 182, 178, 176, 174, 209, 250, 173, 165, 163, 160, 158, 154, 153, 150, 149, 147, 146, 144, 142, 141, 140, 138, 137, 136, 135, 135, 134, 171,  46, 122, 122, 122, 150,  54,  86,  86,  17, 224, 198, 198, 200, 200, 200, 200,
		//199, 199, 198, 198, 196, 196, 194, 194, 192, 191, 190, 189, 186, 185, 183, 181, 179, 174, 172, 223, 255, 255, 215, 161, 159, 156, 155, 151, 149, 148, 146, 144, 142, 141, 139, 138, 137, 135, 134, 133, 132, 132, 132, 169,  46, 120, 119, 119, 147,  54,  84,  84,  17, 224, 193, 193, 194, 195, 196, 195,
		//196, 196, 194, 193, 192, 192, 190, 190, 188, 187, 186, 184, 182, 181, 178, 177, 175, 170, 206, 255, 255, 255, 255, 187, 155, 153, 151, 148, 146, 144, 142, 141, 140, 138, 137, 136, 134, 132, 132, 131, 130, 130, 130, 166,  47, 118, 118, 118, 146,  55,  83,  83,  17, 223, 188, 188, 189, 190, 191, 191,
		//191, 191, 190, 190, 189, 188, 187, 186, 184, 183, 182, 180, 178, 177, 175, 173, 171, 178, 255, 255, 255, 255, 255, 248, 166, 150, 148, 145, 144, 142, 140, 139, 137, 135, 134, 132, 130, 129, 129, 129, 129, 128, 128, 166,  48, 117, 117, 117, 144,  56,  83,  83,  17, 223, 183, 183, 184, 184, 186, 186,
		//186, 186, 186, 185, 184, 183, 182, 181, 179, 177, 177, 175, 174, 171, 170, 168, 166, 220, 255, 255, 255, 255, 255, 255, 235, 152, 144, 141, 137, 134, 131, 128, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 164,  49, 115, 115, 115, 144,  57,  83,  83,  18, 222, 178, 178, 180, 179, 181, 181,
		//181, 181, 180, 180, 180, 178, 207, 235, 173, 173, 171, 176, 168, 166, 165, 163, 172, 255, 255, 255, 255, 255, 255, 255, 255, 233, 133, 127, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 123, 123, 123, 123, 123, 163,  50, 112, 112, 112, 142,  57,  83,  82,  19, 222, 173, 173, 174, 175, 175, 176,
		//175, 175, 175, 175, 173, 204, 255, 248, 169, 168, 205, 242, 163, 161, 159, 157, 193, 255, 255, 255, 255, 255, 255, 255, 255, 255, 205, 138, 147, 122, 121, 121, 121, 121, 121, 120, 120, 120, 120, 120, 120, 120, 119, 161,  49, 110, 110, 110, 140,  57,  81,  81,  20, 221, 166, 168, 168, 168, 170, 170,
		//169, 169, 169, 169, 195, 255, 255, 255, 163, 220, 255, 228, 157, 156, 154, 152, 209, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 175, 236, 143, 117, 117, 117, 117, 117, 116, 116, 116, 116, 116, 116, 115, 115, 160,  48, 106, 106, 106, 138,  55,  79,  79,  20, 219, 160, 161, 161, 162, 162, 164,
		//163, 163, 163, 168, 249, 255, 255, 255, 237, 255, 255, 236, 151, 152, 161, 169, 198, 208, 214, 239, 253, 255, 255, 255, 255, 255, 255, 255, 206, 237, 123, 114, 114, 114, 113, 113, 113, 113, 113, 113, 112, 112, 112, 159,  47, 104, 104, 103, 136,  54,  77,  77,  21, 216, 154, 154, 154, 156, 157, 157,
		//158, 156, 156, 217, 255, 255, 255, 255, 255, 255, 255, 255, 173, 173, 180, 184, 186, 185, 182, 177, 191, 245, 255, 255, 255, 255, 255, 255, 245, 254, 191, 109, 109, 109, 108, 108, 108, 108, 108, 108, 108, 108, 108, 157,  46, 100,  99,  99, 135,  53,  75,  75,  22, 213, 146, 148, 148, 149, 150, 150,
		//150, 151, 157, 255, 255, 255, 255, 255, 255, 255, 255, 255, 250, 188, 178, 183, 187, 188, 186, 181, 175, 179, 241, 255, 255, 255, 255, 255, 255, 254, 246, 115, 106, 106, 106, 105, 105, 105, 105, 105, 105, 105, 105, 157,  44,  97,  97,  96, 134,  51,  73,  73,  23, 211, 142, 142, 142, 143, 143, 144,
		//144, 144, 193, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 250, 209, 176, 179, 182, 184, 187, 241, 239, 229, 255, 255, 255, 255, 255, 255, 255, 254, 160, 102, 102, 102, 102, 102, 102, 101, 101, 101, 101, 101, 156,  43,  93,  93,  93, 134,  51,  72,  71,  24, 207, 134, 134, 134, 136, 136, 136,
		//140, 137, 218, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 245, 203, 173, 176, 176, 174, 223, 255, 255, 255, 255, 255, 255, 255, 255, 254, 197, 100,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99, 156,  42,  91,  91,  91, 134,  49,  70,  70,  25, 203, 129, 128, 130, 129, 130, 130,
		//132, 132, 247, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 249, 163, 167, 168, 167, 170, 243, 255, 255, 255, 255, 255, 255, 255, 254, 225,  97,  97,  96,  96,  96,  96,  96,  96,  96,  96,  96, 158,  41,  88,  88,  88, 135,  48,  70,  70,  27, 200, 122, 123, 123, 123, 124, 125,
		//128, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 228, 170, 154, 158, 160, 160, 158, 193, 255, 255, 255, 255, 255, 255, 255, 254, 255,  95,  95,  95,  95,  95,  94,  94,  94,  94,  94,  93, 157,  40,  87,  87,  87, 135,  47,  69,  69,  28, 196, 118, 118, 118, 119, 119, 121,
		//122, 143, 255, 255, 255, 255, 255, 255, 255, 255, 255, 238, 174, 136, 131, 135, 140, 145, 148, 151, 152, 152, 162, 255, 255, 255, 255, 255, 255, 255, 254, 255,  93,  93,  93,  92,  92,  92,  92,  92,  91,  91,  91, 158,  39,  85,  85,  85, 137,  47,  69,  69,  30, 192, 112, 113, 113, 113, 114, 113,
		//119, 146, 255, 255, 255, 255, 255, 255, 255, 255, 237, 123, 115, 116, 117, 121, 126, 132, 139, 143, 145, 145, 143, 255, 255, 255, 255, 255, 255, 255, 253, 254,  92,  91,  91,  91,  91,  91,  90,  90,  90,  90,  90, 159,  38,  84,  84,  84, 139,  46,  68,  68,  31, 188, 108, 110, 110, 110, 109, 110,
		//114, 108, 250, 255, 255, 255, 255, 255, 255, 255, 180, 105, 104, 104, 105, 109, 114, 120, 127, 134, 138, 138, 137, 255, 255, 255, 255, 254, 255, 255, 253, 254,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  89, 161,  37,  84,  84,  84, 140,  45,  68,  68,  32, 184, 105, 105, 105, 107, 107, 107,
		// 97,  95, 244, 255, 255, 255, 255, 255, 255, 255, 196,  96,  95,  94,  95,  98, 103, 110, 117, 124, 130, 132, 162, 255, 255, 255, 253, 254, 255, 255, 252, 221,  90,  90,  90,  90,  89,  89,  89,  89,  89,  88,  88, 163,  36,  83,  83,  82, 142,  44,  68,  68,  33, 181, 102, 102, 102, 103, 102, 102,
		// 95,  94, 195, 255, 255, 255, 255, 255, 255, 255, 255, 111,  89,  88,  88,  90,  95, 101, 109, 116, 122, 125, 207, 255, 255, 254, 254, 255, 255, 253, 252, 200,  88,  88,  87,  87,  87,  87,  87,  87,  87,  87,  87, 164,  35,  82,  81,  81, 144,  44,  68,  68,  35, 176,  96,  98,  98,  98,  99,  99,
		// 93,  93, 158, 245, 255, 255, 255, 255, 255, 255, 255, 224,  99,  89,  88,  88,  89,  95, 102, 109, 115, 154, 255, 255, 255, 255, 255, 255, 255, 251, 252, 147,  88,  87,  87,  87,  87,  87,  87,  86,  86,  86,  86, 166,  34,  80,  80,  80, 145,  43,  67,  67,  35, 172,  94,  95,  95,  94,  96,  96,
		// 93,  93, 105, 231, 255, 255, 255, 255, 255, 255, 255, 255, 224, 110,  89,  89,  89,  91,  98, 104, 156, 246, 255, 255, 255, 255, 255, 255, 254, 249, 251,  94,  87,  87,  87,  87,  87,  86,  86,  86,  86,  86,  86, 168,  33,  80,  80,  80, 148,  42,  67,  67,  37, 168,  93,  93,  94,  93,  91,  85,
		// 92,  91,  90, 192, 238, 254, 255, 255, 255, 255, 255, 255, 255, 245, 193, 141, 131, 131, 155, 217, 255, 255, 255, 255, 255, 255, 255, 255, 250, 249, 185,  85,  86,  86,  86,  86,  85,  85,  85,  85,  85,  85,  84, 169,  32,  80,  80,  80, 149,  41,  67,  67,  38, 164,  90,  89,  90,  90,  85,  82,
		// 91,  91,  91, 120, 227, 241, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 245, 249, 101,  85,  85,  85,  84,  84,  84,  84,  84,  84,  84,  83,  83, 169,  31,  80,  80,  79, 150,  40,  67,  67,  38, 159,  87,  87,  87,  83,  82,  82,
		// 91,  91,  90,  87, 163, 226, 240, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 243, 246, 162,  84,  85,  85,  84,  84,  84,  84,  84,  84,  84,  83,  83,  83, 171,  30,  80,  79,  79, 152,  40,  67,  66,  39, 156,  86,  86,  84,  82,  82,  81,
		// 90,  90,  90,  90,  93, 199, 225, 237, 251, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 239, 243, 204,  81,  85,  84,  84,  84,  84,  84,  84,  84,  84,  83,  83,  83,  83, 172,  29,  79,  79,  79, 153,  39,  66,  66,  39, 153,  85,  84,  82,  81,  81,  81,
		// 90,  90,  90,  89,  88,  90, 199, 225, 228, 246, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 244, 236, 239, 211,  89,  84,  84,  84,  84,  84,  84,  84,  84,  83,  83,  83,  83,  83,  83, 172,  29,  79,  79,  79, 155,  38,  66,  66,  40, 150,  84,  81,  81,  81,  81,  81,
		// 90,  89,  89,  89,  89,  87,  90, 199, 226, 222, 230, 243, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 245, 235, 232, 235, 208,  88,  84,  84,  84,  84,  84,  84,  84,  83,  83,  83,  83,  83,  83,  83,  82, 173,  28,  79,  79,  78, 155,  38,  66,  66,  40, 146,  81,  81,  81,  81,  81,  81,
		// 89,  89,  89,  89,  88,  88,  87,  90, 162, 228, 225, 222, 223, 229, 233, 242, 242, 243, 244, 236, 231, 226, 227, 230, 233, 155,  77,  84,  84,  84,  84,  84,  84,  84,  83,  83,  83,  83,  83,  83,  82,  82,  82, 173,  27,  79,  78,  78, 156,  37,  66,  66,  39, 143,  81,  81,  81,  81,  81,  81,
		// 89,  89,  89,  88,  88,  88,  87,  87,  83, 115, 191, 228, 226, 224, 223, 222, 222, 222, 223, 225, 226, 228, 230, 173,  95,  80,  84,  84,  84,  84,  84,  84,  83,  83,  83,  83,  83,  83,  83,  82,  82,  82,  82, 173,  27,  78,  78,  78, 156,  37,  65,  65,  39, 141,  81,  81,  81,  81,  80,  80,
		// 89,  88,  88,  88,  88,  87,  87,  87,  87,  83,  79,  96, 143, 191, 229, 229, 228, 229, 229, 219, 191, 142,  95,  79,  83,  84,  84,  84,  84,  84,  84,  83,  83,  83,  83,  83,  83,  83,  82,  82,  82,  82,  82, 173,  26,  78,  78,  78, 156,  36,  65,  65,  39, 139,  81,  81,  80,  80,  80,  80,
	};
	const int imsize=SIZEOF(buffer);
	float src[imsize], dst[imsize];
	for(int k=0;k<imsize;++k)
		src[k]=buffer[k]*(1.f/255);
		//src[k]=k;
	memset(dst, 0, imsize*sizeof(float));
#endif

	const float filt_mag=16;
	float filt[]=
	{
		1, 2, 1,
		2, 4, 2,
		1, 2, 1,

		//-1, -1, -1,//laplacian filter
		//-1,  8, -1,
		//-1, -1, -1,

		// 0, -1,  0,//laplacian filter
		//-1,  4, -1,
		// 0, -1,  0,
	};
	for(int k=0;k<9;++k)
		filt[k]/=filt_mag;
	
#if 1
	ocl_init();
	auto func=k_conv33_const;
	int dim[]={iw, ih};
	size_t worksize[]={imsize};

	long long t1, t2;
	int error=0;
#if 1
	t1=__rdtsc();
	buf_src=p_clCreateBuffer(context, CL_MEM_READ_WRITE, imsize*sizeof(float), nullptr, &error);CL_CHECK(error);
	buf_dim=p_clCreateBuffer(context, CL_MEM_READ_ONLY, 2*sizeof(int), nullptr, &error);		CL_CHECK(error);
	buf_filt=p_clCreateBuffer(context, CL_MEM_READ_WRITE, 9*sizeof(float), nullptr, &error);	CL_CHECK(error);
	buf_dst=p_clCreateBuffer(context, CL_MEM_READ_WRITE, imsize*sizeof(float), nullptr, &error);CL_CHECK(error);
	t2=__rdtsc();
	printf("GPU Alloc:\t%lld\n", t2-t1);
	
	t1=__rdtsc();
	error=p_clEnqueueWriteBuffer(command_queue, buf_src, CL_FALSE, 0, imsize*sizeof(float), src, 0, nullptr, nullptr);	CL_CHECK(error);
	error=p_clEnqueueWriteBuffer(command_queue, buf_dim, CL_FALSE, 0, 2*sizeof(int), dim, 0, nullptr, nullptr);			CL_CHECK(error);
	error=p_clEnqueueWriteBuffer(command_queue, buf_filt, CL_FALSE, 0, 9*sizeof(float), filt, 0, nullptr, nullptr);		CL_CHECK(error);
	error=p_clSetKernelArg(func, 1, sizeof(cl_mem), &buf_dim);	CL_CHECK(error);
	error=p_clSetKernelArg(func, 2, sizeof(cl_mem), &buf_filt);	CL_CHECK(error);
	t2=__rdtsc();
	printf("GPU Send:\t%lld\n", t2-t1);
	t1=__rdtsc();
	for(int k=0;k<5;++k)
	{
		error=p_clSetKernelArg(func, 0, sizeof(cl_mem), &buf_src);	CL_CHECK(error);
		error=p_clSetKernelArg(func, 3, sizeof(cl_mem), &buf_dst);	CL_CHECK(error);
		error=p_clEnqueueNDRangeKernel(command_queue, func, 1, nullptr, worksize, nullptr, 0, nullptr, nullptr);		CL_CHECK(error);
		error=p_clSetKernelArg(func, 0, sizeof(cl_mem), &buf_dst);	CL_CHECK(error);
		error=p_clSetKernelArg(func, 3, sizeof(cl_mem), &buf_src);	CL_CHECK(error);
		error=p_clEnqueueNDRangeKernel(command_queue, func, 1, nullptr, worksize, nullptr, 0, nullptr, nullptr);		CL_CHECK(error);
	}
	error=p_clFlush(command_queue);		CL_CHECK(error);
	error=p_clFinish(command_queue);	CL_CHECK(error);
	t2=__rdtsc();
	printf("Execute:\t%lld\n", t2-t1);
	t1=__rdtsc();
	error=p_clEnqueueReadBuffer(command_queue, buf_dst, CL_TRUE, 0, imsize*sizeof(float), dst, 0, nullptr, nullptr);	CL_CHECK(error);
	t2=__rdtsc();
	printf("GPU Read:\t%lld\n", t2-t1);
#endif
#if 0
	buf_src=p_clCreateBuffer(context, CL_MEM_READ_ONLY, imsize*sizeof(float), nullptr, &error);	CL_CHECK(error);
	buf_dim=p_clCreateBuffer(context, CL_MEM_READ_ONLY, 2*sizeof(int), nullptr, &error);		CL_CHECK(error);
	buf_filt=p_clCreateBuffer(context, CL_MEM_READ_WRITE, 9*sizeof(float), nullptr, &error);	CL_CHECK(error);
	buf_dst=p_clCreateBuffer(context, CL_MEM_WRITE_ONLY, imsize*sizeof(float), nullptr, &error);CL_CHECK(error);

	auto t1=__rdtsc();
	error=p_clEnqueueWriteBuffer(command_queue, buf_src, CL_FALSE, 0, imsize*sizeof(float), src, 0, nullptr, nullptr);	CL_CHECK(error);
	error=p_clEnqueueWriteBuffer(command_queue, buf_dim, CL_FALSE, 0, 2*sizeof(int), dim, 0, nullptr, nullptr);			CL_CHECK(error);
	error=p_clEnqueueWriteBuffer(command_queue, buf_filt, CL_FALSE, 0, 9*sizeof(float), filt, 0, nullptr, nullptr);		CL_CHECK(error);
	error=p_clSetKernelArg(func, 0, sizeof(cl_mem), &buf_src);	CL_CHECK(error);
	error=p_clSetKernelArg(func, 1, sizeof(cl_mem), &buf_dim);	CL_CHECK(error);
	error=p_clSetKernelArg(func, 2, sizeof(cl_mem), &buf_filt);	CL_CHECK(error);
	error=p_clSetKernelArg(func, 3, sizeof(cl_mem), &buf_dst);	CL_CHECK(error);
	error=p_clEnqueueNDRangeKernel(command_queue, func, 1, nullptr, worksize, nullptr, 0, nullptr, nullptr);		CL_CHECK(error);
	error=p_clFlush(command_queue);		CL_CHECK(error);
	error=p_clFinish(command_queue);	CL_CHECK(error);
	error=p_clEnqueueReadBuffer(command_queue, buf_dst, CL_TRUE, 0, imsize*sizeof(float), dst, 0, nullptr, nullptr);CL_CHECK(error);
	auto t2=__rdtsc();
	printf("Cycles: %lld\n", t2-t1);
#endif

	error=p_clReleaseMemObject(buf_src);		CL_CHECK(error);
	error=p_clReleaseMemObject(buf_dim);		CL_CHECK(error);
	error=p_clReleaseMemObject(buf_filt);		CL_CHECK(error);
	error=p_clReleaseMemObject(buf_dst);		CL_CHECK(error);
	ocl_finish();
#endif
#if 0
	for(int k=0;k<10;++k)
	{
		auto t1=__rdtsc();
		conv33_mirror(src, iw, ih, filt, dst);
		auto t2=__rdtsc();
		printf("Cycles: %lld\n", t2-t1);
	}
	//memcpy(dst, src, imsize*sizeof(float));
#endif

	printf("Press any key to overwrite \'a.PNG\'\n");
	_getch();
	printf("Saving...\n");
	std::vector<unsigned char> result(imsize);
	//result.resize(imsize);
	for(int k=0;k<imsize;++k)
		result[k]=(unsigned char)(255*dst[k]);
	//memset(result.data(), 0x7F, imsize);
	lodepng::encode("a.PNG", result, iw, ih, LCT_GREY, 8);
#endif

	exit_success();
	return 0;
}
