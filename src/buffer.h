#pragma once
#ifndef INC_BUFFER_H
#define INC_BUFFER_H
#ifdef __cplusplus
extern "C"
{
#endif

#define		G_BUF_SIZE	4096

extern char
	g_buf[G_BUF_SIZE],
	error_first[G_BUF_SIZE],
	error_latest[G_BUF_SIZE];

#ifdef _WIN32
extern wchar_t g_wbuf[G_BUF_SIZE];
#endif

#ifdef __cplusplus
}
#endif
#endif//INC_BUFFER_H
