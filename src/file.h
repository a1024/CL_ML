#pragma once
#ifndef INC_FILE_H
#define INC_FILE_H
#ifdef __cplusplus
extern "C"
{
#endif
#include"array.h"

int				file_is_readable(const char *filename);//0: not readable, 1: regular file, 2: folder
ArrayHandle		load_text(const char *filename, int pad);
int				save_text(const char *filename, const char *text, size_t len);


#ifdef __cplusplus
}
#endif
#endif//INC_FILE_H
