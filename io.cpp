#include"main.h"
#include<fstream>
#include<iostream>
#include<time.h>
#include<sys/stat.h>
#define	STB_IMAGE_IMPLEMENTATION
#include"stb_image.h"
#include"lodepng.h"
const char		file[]=__FILE__;

#ifndef __GNUC__
#define	S_ISREG(m)	(((m)&S_IFMT)==S_IFREG)
#endif
int				file_is_readable(const char *filename)//0: not readable, 1: regular file, 2: folder
{
#ifdef __GNUC__
	struct stat info={};
	int error=stat(filename, &info);
#else
	struct _stat32 info={};
	int error=_stat32(filename, &info);
#endif
	if(!error)
		return 1+!S_ISREG(info.st_mode);
	return 0;
}

typedef unsigned char byte;
bool			open_text(const char *filename, std::string &data)
{
	std::ifstream input(filename);
	if(!input.is_open())
		return false;
	data.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
	input.close();
	return true;
}
bool			open_bin(const char *filename, std::string &data)
{
	std::ifstream input(filename, std::ios::binary);
	if(!input.is_open())
		return false;
	data.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
	input.close();
	return true;
}
void			load_image(const char *filename, int *&buffer, int &iw, int &ih)
{
	int nch=0;
	byte *original_image=stbi_load(filename, &iw, &ih, &nch, 4);
	if(original_image)
		printf("Opened \'%s\'\n", filename);
	else
	{
		printf("Failed to open \'%s\'\n", filename);
#ifdef _MSC_VER
		scanf_s("%d", &nch);
#endif
		exit(1);
	}
	buffer=(int*)original_image;
}
void			save_image(const char *filename, const int *buffer, int iw, int ih)
{
	printf("About to save result to:\n\n\t\'%s\'\n\nEnter ZERO to save: ", filename);
//	printf("Enter ZERO to save result image: ");
	int x=0;
	scanf_s("%d", &x);
	if(x)
		printf("Didn't save.\n\n");
	else
		lodepng::encode(filename, (const byte*)buffer, iw, ih);//4th overload
}
void			save_image_monochrome(const char *filename, const unsigned char *buffer, int iw, int ih, bool prompt)
{
	if(prompt)
	{
		printf("About to save result to:\n\n\t\'%s\'\n\nEnter ZERO to save: ", filename);
		int x=0;
		scanf_s("%d", &x);
		if(x)
		{
			printf("Didn't save.\n\n");
			return;
		}
	}
	lodepng::encode(filename, (const byte*)buffer, iw, ih, LCT_GREY);//4th overload
}
const char*		gen_filename(double compression_ratio)
{
	time_t t=time(nullptr);
#ifdef __linux__
	auto &now=*localtime(&t);
#else
	tm now={};
	localtime_s(&now, &t);
#endif
	if(compression_ratio)
		sprintf_s(g_buf, G_BUF_SIZE, "%04d%02d%02d_%02d%02d%02d_%g.PNG", 1900+now.tm_year, now.tm_mon+1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, compression_ratio);
	else
		sprintf_s(g_buf, G_BUF_SIZE, "%04d%02d%02d_%02d%02d%02d.PNG", 1900+now.tm_year, now.tm_mon+1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
	return g_buf;
}
void			path_adjust(std::string &path)
{
	for(int k=0;k<(int)path.size();++k)
		if(path[k]=='\\')
			path[k]='/';
	if(path.size()&&path.back()!='/')
		path.push_back('/');
}
void			get_path(std::string &path, const char *format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	std::getline(std::cin, path);
	//clipboard_paste_wide(path);
	path_adjust(path);
	path.back()='\0';
	if(file_is_readable(path.c_str())!=2)
		CRASH("\nNot a directory");
	path.back()='/';
	printf("\n\t%s\n\n", path.c_str());
}


long long		acme_atoi(const char *text, int size, short base, short maxdigits, int *idx, int *ret_ndigits=nullptr)
{
	int ndigits=0;
	byte temp, c;
	long long ival=0;
	int digit_base=base;
	if(digit_base>10)
		digit_base=10;
	if(maxdigits)
		size=*idx+maxdigits;

	for(;*idx<size;++*idx)
	{
		temp=text[*idx];
		if(temp=='\'')
			continue;
		c=temp-'0';
		if(c>=digit_base)
		{
			c=(temp&0xDF)-'A'+10;
			if(c>=base)
				break;
		}
		ival*=base;
		ival+=c;
		++ndigits;
	}
	if(ret_ndigits)
		*ret_ndigits=ndigits;
	return ival;
}
void			skip_ws(const char *text, int size, int *idx)
{
	for(;*idx<size&&iswspace(text[*idx]);++*idx);
}
void			skip_str(const char *text, int size, int *idx, const char *str)
{
	int start=*idx;
	for(;*idx<size&&text[*idx]==str[*idx-start];++*idx);
}
void			load_txt_gains(std::string const &path, WeightInfo const &winfo, std::vector<float> &gains)
{
	printf("Loading  %s  ...\n", winfo.filename);
	std::string fn=path+winfo.filename, str;
	bool success=open_text(fn.c_str(), str);
	MY_ASSERT(success, "Couldn't open  %s\n", fn.c_str());
	auto text=str.c_str();
	int size=(int)str.size(), idx=0;
	
	int s0=(int)gains.size();
	gains.reserve(s0+winfo.nfilt);
	//gains.clear();
	int dst=0, ndigits=0;
	for(int k=0;k<size;)
	{
		idx=k;
		dst=(int)acme_atoi(text, size, 16, 8, &idx, &ndigits);
		MY_ASSERT(ndigits, "Failed to read value.");
		gains.push_back((float&)dst);

	//	printf("\r%3d / %3d: %08X", k+1, winfo.nfilt, dst);//
		
		k+=ndigits;
		skip_ws(text, size, &k);
	}
	MY_ASSERT(gains.size()-s0==winfo.nfilt, "%d != %d\n", (int)gains.size()-s0, winfo.nfilt);
	//printf("\n");
	printf("%s loaded.\n", winfo.filename);
}
void			load_txt_weights_conv(std::string const &path, WeightInfo const &winfo, std::vector<float> &weights)
{
	printf("Loading  %s  ...\n", winfo.filename);
	std::string str=path+winfo.filename;
	bool success=open_text(str.c_str(), str);
	MY_ASSERT(success, "Couldn't open \'%s\'", winfo.filename);
	auto text=str.c_str();
	int size=(int)str.size(), idx=0;

	int nfilters=0, nchannels=0, w=0, h=0;
	nfilters	=(int)acme_atoi(text, size, 10, 0, &idx);
	//if(!nfilters)//
	//	printf("nfilters %d in %s\n%.*s", nfilters, winfo.filename, (int)minimum(size, 200), text);
	skip_ws(text, size, &idx);
	nchannels	=(int)acme_atoi(text, size, 10, 0, &idx);
	skip_ws(text, size, &idx);
	w			=(int)acme_atoi(text, size, 10, 0, &idx);
	skip_ws(text, size, &idx);
	h			=(int)acme_atoi(text, size, 10, 0, &idx);
	skip_ws(text, size, &idx);
	const char errmsg[]="%d != %d\n";
	MY_ASSERT(nfilters==winfo.nfilt, errmsg, nfilters, winfo.nfilt);
	MY_ASSERT(nchannels==winfo.nchan, errmsg, nchannels, winfo.nchan);
	MY_ASSERT(w==winfo.w, errmsg, w, winfo.w);
	MY_ASSERT(h==winfo.h, errmsg, h, winfo.h);

	int s0=(int)weights.size(), totalsize=nfilters*nchannels*w*h;
	weights.resize(s0+totalsize);

	int dst=0, ki=0;
	for(int k=0;k<nfilters;++k)
	{
		skip_str(text, size, &idx, "filt");
		acme_atoi(text, size, 10, 0, &idx);
		skip_ws(text, size, &idx);
		for(int kc=0;kc<nchannels;++kc)
		{
			skip_str(text, size, &idx, "chan");
			acme_atoi(text, size, 10, 0, &idx);
			skip_ws(text, size, &idx);
			for(int ky=0;ky<h;++ky)
			{
				for(int kx=0;kx<w&&idx<size;++kx, ++ki)
				{
					MY_ASSERT(ki<totalsize, "%d >= %d\n", ki, (int)weights.size());
					dst=(int)acme_atoi(text, size, 16, 8, &idx);
					weights[s0+ki]=(float&)dst;
				}
				skip_ws(text, size, &idx);
			}
		}
	}
	MY_ASSERT(ki==totalsize, errmsg, ki, (int)weights.size());
	printf("%s loaded.\n", winfo.filename);
}
void			load_txt_weights_fc(std::string const &path, WeightInfo const &winfo, std::vector<float> &weights)
{
	printf("Loading  %s  ...\n", winfo.filename);
	std::string str=path+winfo.filename;
	bool success=open_text(str.c_str(), str);
	MY_ASSERT(success, "Couldn't open \'%s\'", winfo.filename);
	auto text=str.c_str();
	int size=(int)str.size(), idx=0;

	int ninputs=0, noutputs=0;
	noutputs=(int)acme_atoi(text, size, 10, 0, &idx);
	skip_ws(text, size, &idx);
	ninputs	=(int)acme_atoi(text, size, 10, 0, &idx);
	skip_ws(text, size, &idx);
	const char errmsg[]="%d != %d\n";
	MY_ASSERT(noutputs==winfo.nfilt, errmsg, noutputs, winfo.nfilt);
	MY_ASSERT(ninputs==winfo.nchan, errmsg, ninputs, winfo.nchan);

	int totalsize=noutputs*ninputs;
	weights.resize(totalsize);

	int dst=0, kcoeff=0;
	for(int ko=0;ko<noutputs;++ko)
	{
		for(int ki=0;ki<ninputs;++ki, ++kcoeff)
		{
			MY_ASSERT(ki<totalsize, "%d >= %d\n", kcoeff, (int)weights.size());
			dst=(int)acme_atoi(text, size, 16, 8, &idx);
			weights[kcoeff]=(float&)dst;
		}
		skip_ws(text, size, &idx);
	}
	MY_ASSERT(kcoeff==totalsize, errmsg, kcoeff, (int)weights.size());
	printf("%s loaded.\n", winfo.filename);
}


const int		magic_conv='C'|'O'<<8|'N'<<16|'V'<<24;
const int		magic_linr='L'|'I'<<8|'N'<<16|'R'<<24;
struct			ConvFileHeader
{
	union
	{
		char tag[4];//conv_magic
		int magic;
	};
	short nfilters, nch, kw, kh, pad, stride;
	float data[];
};
bool			save_weights_bin(const float *weights, int count, const char *filename, WeightInfo const &info, BinFileType type)
{
	ConvFileHeader header;
	switch(type)
	{
	case BINFILE_CONV:
		header.magic=magic_conv;
		break;
	case BINFILE_LINEAR:
		header.magic=magic_linr;
		break;
	default:
		return false;
	}
	header.nfilters=info.nfilt;
	header.nch=info.nchan;
	header.kw=info.w;
	header.kh=info.h;
	header.pad=info.pad;
	header.stride=info.stride;

	std::ofstream out(filename, std::ios::out|std::ios::binary);
	if(!out.is_open())
		return false;
	out.write((const char*)&header, sizeof(header));
	out.write((const char*)weights, count*sizeof(float));
	out.close();
	return true;
}
void			load_weights_bin(std::string const &path, const char *filename, WeightInfo const &winfo, std::vector<float> &weights)
{
	printf("Loading  %s...\n", filename);
	std::string str=path+filename;
	bool success=open_bin(str.c_str(), str);
	MY_ASSERT(success, "Couldn't open \'%s\'", filename);
	int filesize=(int)str.size(), idx=0;
	MY_ASSERT(filesize>=sizeof(ConvFileHeader), "Wrong file size.\n");
	auto data=str.c_str();
	ConvFileHeader header;
	memcpy(&header, data, sizeof(header));
	MY_ASSERT(header.magic==magic_conv||header.magic==magic_linr, "Wrong file %s", filename);
	auto totalsize=str.size()-sizeof(ConvFileHeader);
	MY_ASSERT(!(totalsize&3), "%s has wrong size of %d, which is not divisible by sizeof(float)", filename, (int)str.size());
	totalsize/=sizeof(float);
	weights.resize(totalsize);
	memcpy(weights.data(), data+sizeof(ConvFileHeader), totalsize*sizeof(float));
	printf("%s loaded.\n", filename);
}
