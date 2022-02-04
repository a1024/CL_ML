#pragma once
#ifndef MAIN_H
#define MAIN_H
#include<vector>
#include<string>
#include"runtime.h"

//I/O operations
int				file_is_readable(const char *filename);//0: not readable, 1: regular file, 2: folder
bool			open_text(const char *filename, std::string &data);
void			load_image(const char *filename, int *&buffer, int &iw, int &ih);
void			save_image(const char *filename, const int *buffer, int iw, int ih);
void			save_image_monochrome(const char *filename, const unsigned char *buffer, int iw, int ih, bool prompt);
const char*		gen_filename(double compression_ratio=0);
void			path_adjust(std::string &path);
void			get_path(std::string &path, const char *format, ...);

struct			WeightInfo
{
	short nfilt, nchan;
	unsigned char w, h, pad, stride;
	const char *filename;
};
void			load_txt_gains(std::string const &path, WeightInfo const &winfo, std::vector<float> &gains);
void			load_txt_weights_conv(std::string const &path, WeightInfo const &winfo, std::vector<float> &weights);
void			load_txt_weights_fc(std::string const &path, WeightInfo const &winfo, std::vector<float> &weights);

enum			BinFileType
{
	BINFILE_CONV,
	BINFILE_LINEAR,
};
bool			save_weights_bin(const float *weights, int count, const char *filename, WeightInfo const &info, BinFileType type);
void			load_weights_bin(std::string const &path, const char *filename, WeightInfo const &winfo, std::vector<float> &weights);


//imagenet1000 classes
struct			ResultClass
{
	int idx;
	const char *name;
};
extern ResultClass classtable[1000];


//Entropy Coders
void			abac2_encode(const void *src, int imsize, int depth, int bytestride, std::string &out_data, int *out_sizes, int *out_conf, bool loud);
void			abac2_decode(const char *data, const int *sizes, const int *conf, void *dst, int imsize, int depth, int bytestride, bool loud);

int				abac2_estimate(const void *src, int imsize, int depth, int bytestride, int *out_sizes, int *out_conf, bool loud);

enum			WeightFileLabel
{
#define			WNAME(NFILT, NCHAN, WIDTH, HEIGHT, PAD, STRIDE, FILENAME, LABEL)		LABEL,
#include		"names_resnet18.h"
#undef			WNAME
};

#endif
