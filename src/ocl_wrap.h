#pragma once
#ifndef INC_OCL_WRAP_H
#define INC_OCL_WRAP_H
#ifdef __cplusplus
extern "C"
{
#endif
#define	CL_TARGET_OPENCL_VERSION	120
#include<CL/opencl.h>


#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4229)//anachronism used : modifiers on data are ignored
#endif
typedef cl_int			(*CL_API_CALL t_clGetPlatformIDs)(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
typedef cl_int			(*CL_API_CALL t_clGetPlatformInfo)(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
typedef cl_int			(*CL_API_CALL t_clGetDeviceIDs)(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
typedef cl_int			(*CL_API_CALL t_clGetDeviceInfo)(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
typedef cl_context		(*CL_API_CALL t_clCreateContext)(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (CL_CALLBACK *pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data), void *user_data, cl_int *errcode_ret);
typedef cl_int			(*CL_API_CALL t_clReleaseContext)(cl_context context);
typedef cl_int			(*CL_API_CALL t_clRetainContext)(cl_context context);
typedef cl_int			(*CL_API_CALL t_clGetContextInfo)(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
typedef cl_command_queue(*CL_API_CALL t_clCreateCommandQueue)(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret);
typedef cl_program		(*CL_API_CALL t_clCreateProgramWithSource)(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
typedef cl_int			(*CL_API_CALL t_clBuildProgram)(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data);
typedef cl_int			(*CL_API_CALL t_clGetProgramBuildInfo)(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
typedef cl_int			(*CL_API_CALL t_clGetProgramInfo)(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
typedef cl_program		(*CL_API_CALL t_clCreateProgramWithBinary)(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret);
typedef cl_mem			(*CL_API_CALL t_clCreateBuffer)(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
typedef cl_kernel		(*CL_API_CALL t_clCreateKernel)(cl_program program, const char *kernel_name, cl_int *errcode_ret);
typedef cl_int			(*CL_API_CALL t_clSetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
typedef cl_int			(*CL_API_CALL t_clEnqueueFillBuffer)(cl_command_queue command_queue, cl_mem buffer, const void * pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
typedef cl_int			(*CL_API_CALL t_clEnqueueWriteBuffer)(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
typedef cl_int			(*CL_API_CALL t_clEnqueueCopyBuffer)(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
typedef cl_int			(*CL_API_CALL t_clEnqueueNDRangeKernel)(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t * global_work_offset, const size_t * global_work_size, const size_t * local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
typedef cl_int			(*CL_API_CALL t_clEnqueueReadBuffer)(cl_command_queue command_queue, cl_mem  buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
typedef cl_int			(*CL_API_CALL t_clFlush)(cl_command_queue command_queue);
typedef cl_int			(*CL_API_CALL t_clFinish)(cl_command_queue command_queue);
typedef cl_mem			(*CL_API_CALL t_clCreateFromGLBuffer)(cl_context context, cl_mem_flags flags, cl_GLuint bufobj, cl_int *errcode_ret);
typedef cl_mem			(*CL_API_CALL t_clCreateFromGLTexture)(cl_context context, cl_mem_flags flags, cl_GLenum target, cl_GLint miplevel, cl_GLuint texture, cl_int *errcode_ret);
typedef cl_int			(*CL_API_CALL t_clReleaseMemObject)(cl_mem memobj);
typedef cl_int			(*CL_API_CALL t_clReleaseKernel)(cl_kernel kernel);
typedef cl_int			(*CL_API_CALL t_clReleaseProgram)(cl_program program);
typedef cl_int			(*CL_API_CALL t_clReleaseCommandQueue)(cl_command_queue command_queue);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define OPENCL_FUNC(FUNCNAME)	extern t_##FUNCNAME p_##FUNCNAME
#include"ocl_wrap_func.h"
#undef	OPENCL_FUNC

int oclw_loadAPI();
const char*	clerr2str(int error);
#define		CL_CHECK(ERR)		ASSERT_MSG(!(ERR), "OpenCL error %d: %s\n", ERR, clerr2str(ERR))


#ifdef __cplusplus
}
#endif
#endif//INC_OCL_WRAP_H
