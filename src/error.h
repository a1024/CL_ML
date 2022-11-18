#pragma once
#ifndef INC_ERROR_H
#define INC_ERROR_H
#ifdef __cplusplus
extern "C"
{
#endif

	
void	pause();
int		valid(const void *ptr);
int		log_error(const char *file, int line, int quit, const char *format, ...);
#define	LOG_WARNING(format, ...)			log_error(file, __LINE__, 0, format, ##__VA_ARGS__)
#define	LOG_ERROR(format, ...)				log_error(file, __LINE__, 1, format, ##__VA_ARGS__)
#define	ASSERT_MSG(SUCCESS, format, ...)	((SUCCESS)!=0||log_error(file, __LINE__, 1, format, ##__VA_ARGS__))
#define ASSERT_PTR(PTR)						(valid(PTR)||log_error(file, __LINE__, 1, #PTR " == NULL"))


#ifdef __cplusplus
}
#endif
#endif//INC_ERROR_H
