#include<Windows.h>
#include<GL/gl.h>
#include<stdio.h>
#include"lodepng.h"
#include"error.h"
#include"glml.h"
#include"buffer.h"
#include"file.h"
#ifdef _MSC_VER
#pragma comment(lib, "OpenGL32.lib")
#endif
static const char file[]=__FILE__;

#ifdef _MSC_VER
HWND ghWnd=0;
HDC ghDC=0;
HGLRC hRC=0;
#endif

const char *GLversion=0;

#define GLFUNC(X)	t_##X X=0;
GLFUNCLIST
#undef	GLFUNC

unsigned programID=0;
const char programname[]="hello_compute";

unsigned txid=0;


int acme_timestamp(char *buf, size_t len);

void	copy_to_clipboard_c(const char *a, int size)//size not including null terminator
{
	char *clipboard=(char*)LocalAlloc(LMEM_FIXED, (size+1)*sizeof(char));
	memcpy(clipboard, a, (size+1)*sizeof(char));
	clipboard[size]='\0';
	OpenClipboard(0);
	EmptyClipboard();
	SetClipboardData(CF_OEMTEXT, (void*)clipboard);
	CloseClipboard();
}
#if 0
char*	paste_from_clipboard(int loud, int *ret_len)
{
	OpenClipboard(0);
	char *a=(char*)GetClipboardData(CF_OEMTEXT);
	if(!a)
	{
		CloseClipboard();
		if(loud)
			messagebox(MBOX_OK, "Error", "Failed to paste from clipboard");
		return 0;
	}
	int len0=strlen(a);

	char *str=(char*)malloc(len0+1);
	if(!str)
		LOG_ERROR("paste_from_clipboard: malloc(%d) returned 0", len0+1);
	int len=0;
	for(int k2=0;k2<len0;++k2)
	{
		if(a[k2]!='\r')
			str[len]=a[k2], ++len;
	}
	str[len]='\0';

	CloseClipboard();
	if(ret_len)
		*ret_len=len;
	return str;
}
#endif

const char*	glerr2str(int error)
{
#define 			EC(x)	case x:a=(const char*)#x;break
	const char *a=0;
	switch(error)
	{
	case 0:a="SUCCESS";break;
	EC(GL_INVALID_ENUM);
	EC(GL_INVALID_VALUE);
	EC(GL_INVALID_OPERATION);
	case 0x0503:a="GL_STACK_OVERFLOW";break;
	case 0x0504:a="GL_STACK_UNDERFLOW";break;
	EC(GL_OUT_OF_MEMORY);
	case 0x0506:a="GL_INVALID_FRAMEBUFFER_OPERATION";break;
	case 0x0507:a="GL_CONTEXT_LOST";break;
	case 0x8031:a="GL_TABLE_TOO_LARGE";break;
	default:a="???";break;
	}
	return a;
#undef				EC
}
int			sys_check(const char *file, int line, const char *info)
{
	int error=GetLastError();
	if(error)
	{
		char *messageBuffer=0;
		size_t size=FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS, NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
		log_error(file, line, 0, "%s%sGetLastError() returned %d: %s", info?info:"", info?"\n":"", error, messageBuffer);
		LocalFree(messageBuffer);
	}
	return 0;
}
static int gl_loaded=0;

unsigned CompileShader(const char *src, size_t srclen, unsigned type, const char *programname)
{
	int error;
	unsigned shaderID=glCreateShader(type);
	glShaderSource(shaderID, 1, &src, 0);
	glCompileShader(shaderID);
	int success=0;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
	if(!success)
	{
		int infoLogLength;
		glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
		char *errorMessage=(char*)malloc(infoLogLength+1);
		glGetShaderInfoLog(shaderID, infoLogLength, 0, errorMessage);
		copy_to_clipboard_c(errorMessage, infoLogLength);
		if(programname)
			LOG_ERROR("%s shader compilation failed. Output copied to clipboard.", programname);
		else
			GL_ERROR(error);
		free(errorMessage);
		return 0;
	}
	return shaderID;
}
LRESULT __stdcall WndProc(HWND hWnd, unsigned message, WPARAM wParam, LPARAM lParam)
{
	switch(message)
	{
	case WM_CREATE:
		break;
	}
	return DefWindowProcA(hWnd, message, wParam, lParam);
}
void init_gl()
{
	WNDCLASSEXA wndClassEx=
	{
		sizeof(WNDCLASSEXA), CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS,
		WndProc, 0, 0, 0,
		LoadIconA(0, (char*)0x00007F00),
		LoadCursorA(0, (char*)0x00007F00),
		0,
		0, "New format", 0
	};
	int error;
#ifdef _MSC_VER
	if(!gl_loaded)
	{
		error=RegisterClassExA(&wndClassEx);	ASSERT_MSG(error, "Failed to register window class");
		ghWnd=CreateWindowExA(0, wndClassEx.lpszClassName, "", 0, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, 0, 0, 0, 0);	ASSERT_MSG(ghWnd, "CreateWindow returned NULL");
		ghDC=GetDC(ghWnd);
		PIXELFORMATDESCRIPTOR pfd=
		{
			sizeof(PIXELFORMATDESCRIPTOR), 1,
			PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,
			PFD_TYPE_RGBA, 32,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			16,//depth bits
			0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
		};
		int pixelformat=ChoosePixelFormat(ghDC, &pfd);
		SetPixelFormat(ghDC, pixelformat, &pfd);
		hRC=wglCreateContext(ghDC);		ASSERT_MSG(hRC, "wglCreateContext returned NULL");
		wglMakeCurrent(ghDC, hRC);

		GLversion=(const char*)glGetString(GL_VERSION);

		error=0;
#define GLFUNC(X)		(PROC)X=wglGetProcAddress(#X), (X!=0||(error+=!sys_check(file, __LINE__, #X " == nullptr")));
GLFUNCLIST
#undef	GLFUNC
		ASSERT_MSG(!error, "Failed to load %d OpenGL function(s)", error);
		gl_loaded=1;
	}
#endif

	//compile shader
	ArrayHandle src=load_file("gl_shader.h", 0, 0);
	unsigned shaderID=CompileShader(src->data, src->count, GL_COMPUTE_SHADER, programname);
	programID=glCreateProgram();
	glAttachShader(programID, shaderID);
	glLinkProgram(programID);
	
	int success=0;
	glGetProgramiv(programID, GL_LINK_STATUS, &success);
	if(!success)
	{
		int infoLogLength;
		glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
		char *errorMessage=(char*)malloc(infoLogLength+1);
		glGetProgramInfoLog(programID, infoLogLength, 0, errorMessage);
		copy_to_clipboard_c(errorMessage, infoLogLength);
		if(programname)
			LOG_ERROR("%s shader link failed. Output copied to cipboard.", programname);
		else
			GL_ERROR(error);
		free(errorMessage);
		//return 0;
	}
	glDetachShader(programID, shaderID);
	glDeleteShader(shaderID);


	//gen texture
	int iw=128, ih=128, size=iw*ih;
	glGenTextures(1, &txid);				GL_CHECK(error);//glCreateTexture
	glBindTexture(GL_TEXTURE_2D, txid);		GL_CHECK(error);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);		GL_CHECK(error);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);		GL_CHECK(error);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, iw, ih);					GL_CHECK(error);
	glBindImageTexture(0, txid, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);	GL_CHECK(error);


	//dispatch compute
	glUseProgram(programID);				GL_CHECK(error);
	glDispatchCompute(iw/8, ih/4, 1);		GL_CHECK(error);
	glMemoryBarrier(GL_ALL_BARRIER_BITS);	GL_CHECK(error);


	//get results	//https://stackoverflow.com/questions/5117653/how-to-get-texture-data-using-textureids-in-opengl/62965713#62965713
	unsigned PBO=0;
	glGenBuffers(1, &PBO);					GL_CHECK(error);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, PBO);GL_CHECK(error);
	glBufferData(GL_PIXEL_PACK_BUFFER, size*sizeof(float), 0, GL_STATIC_READ);	GL_CHECK(error);

	glBindTexture(GL_TEXTURE_2D, txid);		GL_CHECK(error);//get texture image
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);	GL_CHECK(error);
	unsigned char *result=(unsigned char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);	GL_CHECK(error);

	//int *image=(int*)malloc(size*sizeof(int));
	//for(int k=0;k<size*sizeof(int);++k)
	//{
	//	float val=255*result[k];
	//	if(val<0)
	//		val=0;
	//	if(val>255)
	//		val=255;
	//	((unsigned char*)image)[k]=(unsigned char)val;
	//}
	int printed=acme_timestamp(g_buf, G_BUF_SIZE);
	printed+=sprintf_s(g_buf+printed, G_BUF_SIZE-printed, ".PNG");
	lodepng_encode_file(g_buf, result, iw, ih, LCT_RGBA, 8);

	glUnmapBuffer(GL_PIXEL_PACK_BUFFER);	GL_CHECK(error);

	pause();
	exit(0);
}