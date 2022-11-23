#include"acme_stdio.h"
#include<string.h>
#include<sys/stat.h>
#include<errno.h>
#include"array.h"
#include"error.h"
static const char file[]=__FILE__;

#if !defined __linux__
#if defined _MSC_VER && _MSC_VER<1800
#define	S_IFMT		00170000//octal
#define	S_IFREG		 0100000
#endif
#ifndef S_ISREG
#define	S_ISREG(m)	(((m)&S_IFMT)==S_IFREG)
#endif
#endif
int				file_is_readable(const char *filename)//0: not readable, 1: regular file, 2: folder
{
	struct stat info={0};

	int error=stat(filename, &info);
	if(!error)
		return 1+!S_ISREG(info.st_mode);
	return 0;
}
ArrayHandle		load_text(const char *filename, int pad)
{
	struct stat info={0};
	FILE *f;
	ArrayHandle str;

	int error=stat(filename, &info);
	if(error)
	{
		LOG_ERROR("Cannot open %s\n%s", filename, strerror(errno));
		return 0;
	}
	f=fopen(filename, "r");

	str=array_construct(0, 1, info.st_size, 1, pad+1, 0);
	str->count=fread(str->data, 1, info.st_size, f);
	fclose(f);
	memset(str->data+str->count, 0, str->cap-str->count);
	return str;
}
int				save_text(const char *filename, const char *text, size_t len)
{
	FILE *f=fopen(filename, "w");
	if(!f)
		return 0;
	fwrite(text, 1, len, f);
	fclose(f);
	return 1;
}