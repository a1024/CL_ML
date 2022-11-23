#include"acme_stdio.h"
#include<stddef.h>
#include<stdarg.h>
#include<stdlib.h>
#include<string.h>
#include"buffer.h"
#include"error.h"

char
	g_buf[G_BUF_SIZE]={0},
	error_first[G_BUF_SIZE]={0},
	error_latest[G_BUF_SIZE]={0};

#ifdef _WIN32
wchar_t g_wbuf[G_BUF_SIZE]={0};
#endif

void	pause()
{
	int k, count;

	printf("Enter 0 to continue: ");
	count=scanf("%d", &k);
}
int		valid(const void *ptr)
{
	size_t val=(size_t)ptr;

	if(sizeof(size_t)==4)
	{
		switch(val)
		{
		case 0:
		case 0xCCCCCCCC:
		case 0xFEEEFEEE:
		case 0xEEFEEEFE:
		case 0xCDCDCDCD:
		case 0xFDFDFDFD:
		case 0xBAADF00D:
		case 0xBAAD0000:
			return 0;
		}
	}
	else
	{
		if(val==0xCCCCCCCCCCCCCCCC)
			return 0;
		if(val==0xFEEEFEEEFEEEFEEE)
			return 0;
		if(val==0xEEFEEEFEEEFEEEFE)
			return 0;
		if(val==0xCDCDCDCDCDCDCDCD)
			return 0;
		if(val==0xBAADF00DBAADF00D)
			return 0;
		if(val==0xADF00DBAADF00DBA)
			return 0;
	}
	return 1;
}
int		log_error(const char *file, int line, int quit, const char *format, ...)
{
	int firsttime=error_first[0]=='\0';

	ptrdiff_t size=(ptrdiff_t)strlen(file), start=size-1;
	for(;start>=0&&file[start]!='/'&&file[start]!='\\';--start);
	start+=start==-1||file[start]=='/'||file[start]=='\\';

	int printed=sprintf_s(error_latest, G_BUF_SIZE, "%s(%d): ", file+start, line);
	va_list args;
	va_start(args, format);
	printed+=vsprintf_s(error_latest+printed, G_BUF_SIZE-printed, format, args);
	va_end(args);

	if(firsttime)
		memcpy(error_first, error_latest, printed+1);
#ifdef HAVE_GUI
	messagebox(MBOX_OK, "Error", error_latest);
#else
	fprintf(stderr, "%s\n", error_latest);
#endif
	if(quit)
	{
		pause();
		exit(1);
	}
	return 0;
}