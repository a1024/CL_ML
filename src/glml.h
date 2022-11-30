#pragma once
#ifndef INC_GLML_H
#define INC_GLML_H
#ifdef __cplusplus
extern "C"
{
#endif
	
	
#ifdef _MSC_VER
#define				GL_FUNC_ADD				0x8006//GL/glew.h
#define				GL_MIN					0x8007
#define				GL_MAX					0x8008
#define				GL_MAJOR_VERSION		0x821B
#define				GL_MINOR_VERSION		0x821C
#define				GL_TEXTURE0				0x84C0
#define				GL_TEXTURE1				0x84C1
#define				GL_TEXTURE2				0x84C2
#define				GL_TEXTURE3				0x84C3
#define				GL_TEXTURE4				0x84C4
#define				GL_TEXTURE5				0x84C5
#define				GL_TEXTURE6				0x84C6
#define				GL_TEXTURE7				0x84C7
#define				GL_TEXTURE8				0x84C8
#define				GL_TEXTURE9				0x84C9
#define				GL_TEXTURE10			0x84CA
#define				GL_TEXTURE11			0x84CB
#define				GL_TEXTURE12			0x84CC
#define				GL_TEXTURE13			0x84CD
#define				GL_TEXTURE14			0x84CE
#define				GL_TEXTURE15			0x84CF
#define				GL_TEXTURE_RECTANGLE	0x84F5
#define				GL_PROGRAM_POINT_SIZE	0x8642
#define				GL_BUFFER_SIZE			0x8764
#define				GL_ARRAY_BUFFER			0x8892
#define				GL_ELEMENT_ARRAY_BUFFER	0x8893
#define				GL_STATIC_DRAW			0x88E4
#define				GL_FRAGMENT_SHADER		0x8B30
#define				GL_VERTEX_SHADER		0x8B31
#define				GL_COMPILE_STATUS		0x8B81
#define				GL_LINK_STATUS			0x8B82
#define				GL_INFO_LOG_LENGTH		0x8B84
#define				GL_DEBUG_OUTPUT			0x92E0//OpenGL 4.3+
	
#define GL_CURRENT_RASTER_SECONDARY_COLOR 0x845F
#define GL_PIXEL_PACK_BUFFER 0x88EB
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#define GL_PIXEL_PACK_BUFFER_BINDING 0x88ED
#define GL_PIXEL_UNPACK_BUFFER_BINDING 0x88EF
#define GL_FLOAT_MAT2x3 0x8B65
#define GL_FLOAT_MAT2x4 0x8B66
#define GL_FLOAT_MAT3x2 0x8B67
#define GL_FLOAT_MAT3x4 0x8B68
#define GL_FLOAT_MAT4x2 0x8B69
#define GL_FLOAT_MAT4x3 0x8B6A
#define GL_SRGB 0x8C40
#define GL_SRGB8 0x8C41
#define GL_SRGB_ALPHA 0x8C42
#define GL_SRGB8_ALPHA8 0x8C43
#define GL_SLUMINANCE_ALPHA 0x8C44
#define GL_SLUMINANCE8_ALPHA8 0x8C45
#define GL_SLUMINANCE 0x8C46
#define GL_SLUMINANCE8 0x8C47
#define GL_COMPRESSED_SRGB 0x8C48
#define GL_COMPRESSED_SRGB_ALPHA 0x8C49
#define GL_COMPRESSED_SLUMINANCE 0x8C4A
#define GL_COMPRESSED_SLUMINANCE_ALPHA 0x8C4B
	
#define GL_CLIP_DISTANCE0 GL_CLIP_PLANE0
#define GL_CLIP_DISTANCE1 GL_CLIP_PLANE1
#define GL_CLIP_DISTANCE2 GL_CLIP_PLANE2
#define GL_CLIP_DISTANCE3 GL_CLIP_PLANE3
#define GL_CLIP_DISTANCE4 GL_CLIP_PLANE4
#define GL_CLIP_DISTANCE5 GL_CLIP_PLANE5
#define GL_COMPARE_REF_TO_TEXTURE GL_COMPARE_R_TO_TEXTURE_ARB
#define GL_MAX_CLIP_DISTANCES GL_MAX_CLIP_PLANES
#define GL_MAX_VARYING_COMPONENTS GL_MAX_VARYING_FLOATS
#define GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT 0x0001
#define GL_MAJOR_VERSION 0x821B
#define GL_MINOR_VERSION 0x821C
#define GL_NUM_EXTENSIONS 0x821D
#define GL_CONTEXT_FLAGS 0x821E
#define GL_DEPTH_BUFFER 0x8223
#define GL_STENCIL_BUFFER 0x8224
#define GL_RGBA32F 0x8814
#define GL_RGB32F 0x8815
#define GL_RGBA16F 0x881A
#define GL_RGB16F 0x881B
#define GL_VERTEX_ATTRIB_ARRAY_INTEGER 0x88FD
#define GL_MAX_ARRAY_TEXTURE_LAYERS 0x88FF
#define GL_MIN_PROGRAM_TEXEL_OFFSET 0x8904
#define GL_MAX_PROGRAM_TEXEL_OFFSET 0x8905
#define GL_CLAMP_VERTEX_COLOR 0x891A
#define GL_CLAMP_FRAGMENT_COLOR 0x891B
#define GL_CLAMP_READ_COLOR 0x891C
#define GL_FIXED_ONLY 0x891D
#define GL_TEXTURE_RED_TYPE 0x8C10
#define GL_TEXTURE_GREEN_TYPE 0x8C11
#define GL_TEXTURE_BLUE_TYPE 0x8C12
#define GL_TEXTURE_ALPHA_TYPE 0x8C13
#define GL_TEXTURE_LUMINANCE_TYPE 0x8C14
#define GL_TEXTURE_INTENSITY_TYPE 0x8C15
#define GL_TEXTURE_DEPTH_TYPE 0x8C16
#define GL_TEXTURE_1D_ARRAY 0x8C18
#define GL_PROXY_TEXTURE_1D_ARRAY 0x8C19
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#define GL_PROXY_TEXTURE_2D_ARRAY 0x8C1B
#define GL_TEXTURE_BINDING_1D_ARRAY 0x8C1C
#define GL_TEXTURE_BINDING_2D_ARRAY 0x8C1D
#define GL_R11F_G11F_B10F 0x8C3A
#define GL_UNSIGNED_INT_10F_11F_11F_REV 0x8C3B
#define GL_RGB9_E5 0x8C3D
#define GL_UNSIGNED_INT_5_9_9_9_REV 0x8C3E
#define GL_TEXTURE_SHARED_SIZE 0x8C3F
#define GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH 0x8C76
#define GL_TRANSFORM_FEEDBACK_BUFFER_MODE 0x8C7F
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS 0x8C80
#define GL_TRANSFORM_FEEDBACK_VARYINGS 0x8C83
#define GL_TRANSFORM_FEEDBACK_BUFFER_START 0x8C84
#define GL_TRANSFORM_FEEDBACK_BUFFER_SIZE 0x8C85
#define GL_PRIMITIVES_GENERATED 0x8C87
#define GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN 0x8C88
#define GL_RASTERIZER_DISCARD 0x8C89
#define GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS 0x8C8A
#define GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS 0x8C8B
#define GL_INTERLEAVED_ATTRIBS 0x8C8C
#define GL_SEPARATE_ATTRIBS 0x8C8D
#define GL_TRANSFORM_FEEDBACK_BUFFER 0x8C8E
#define GL_TRANSFORM_FEEDBACK_BUFFER_BINDING 0x8C8F
#define GL_RGBA32UI 0x8D70
#define GL_RGB32UI 0x8D71
#define GL_RGBA16UI 0x8D76
#define GL_RGB16UI 0x8D77
#define GL_RGBA8UI 0x8D7C
#define GL_RGB8UI 0x8D7D
#define GL_RGBA32I 0x8D82
#define GL_RGB32I 0x8D83
#define GL_RGBA16I 0x8D88
#define GL_RGB16I 0x8D89
#define GL_RGBA8I 0x8D8E
#define GL_RGB8I 0x8D8F
#define GL_RED_INTEGER 0x8D94
#define GL_GREEN_INTEGER 0x8D95
#define GL_BLUE_INTEGER 0x8D96
#define GL_ALPHA_INTEGER 0x8D97
#define GL_RGB_INTEGER 0x8D98
#define GL_RGBA_INTEGER 0x8D99
#define GL_BGR_INTEGER 0x8D9A
#define GL_BGRA_INTEGER 0x8D9B
#define GL_SAMPLER_1D_ARRAY 0x8DC0
#define GL_SAMPLER_2D_ARRAY 0x8DC1
#define GL_SAMPLER_1D_ARRAY_SHADOW 0x8DC3
#define GL_SAMPLER_2D_ARRAY_SHADOW 0x8DC4
#define GL_SAMPLER_CUBE_SHADOW 0x8DC5
#define GL_UNSIGNED_INT_VEC2 0x8DC6
#define GL_UNSIGNED_INT_VEC3 0x8DC7
#define GL_UNSIGNED_INT_VEC4 0x8DC8
#define GL_INT_SAMPLER_1D 0x8DC9
#define GL_INT_SAMPLER_2D 0x8DCA
#define GL_INT_SAMPLER_3D 0x8DCB
#define GL_INT_SAMPLER_CUBE 0x8DCC
#define GL_INT_SAMPLER_1D_ARRAY 0x8DCE
#define GL_INT_SAMPLER_2D_ARRAY 0x8DCF
#define GL_UNSIGNED_INT_SAMPLER_1D 0x8DD1
#define GL_UNSIGNED_INT_SAMPLER_2D 0x8DD2
#define GL_UNSIGNED_INT_SAMPLER_3D 0x8DD3
#define GL_UNSIGNED_INT_SAMPLER_CUBE 0x8DD4
#define GL_UNSIGNED_INT_SAMPLER_1D_ARRAY 0x8DD6
#define GL_UNSIGNED_INT_SAMPLER_2D_ARRAY 0x8DD7
#define GL_QUERY_WAIT 0x8E13
#define GL_QUERY_NO_WAIT 0x8E14
#define GL_QUERY_BY_REGION_WAIT 0x8E15
#define GL_QUERY_BY_REGION_NO_WAIT 0x8E16
	
#define GL_CURRENT_FOG_COORD GL_CURRENT_FOG_COORDINATE
#define GL_FOG_COORD GL_FOG_COORDINATE
#define GL_FOG_COORD_ARRAY GL_FOG_COORDINATE_ARRAY
#define GL_FOG_COORD_ARRAY_BUFFER_BINDING GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING
#define GL_FOG_COORD_ARRAY_POINTER GL_FOG_COORDINATE_ARRAY_POINTER
#define GL_FOG_COORD_ARRAY_STRIDE GL_FOG_COORDINATE_ARRAY_STRIDE
#define GL_FOG_COORD_ARRAY_TYPE GL_FOG_COORDINATE_ARRAY_TYPE
#define GL_FOG_COORD_SRC GL_FOG_COORDINATE_SOURCE
#define GL_SRC0_ALPHA GL_SOURCE0_ALPHA
#define GL_SRC0_RGB GL_SOURCE0_RGB
#define GL_SRC1_ALPHA GL_SOURCE1_ALPHA
#define GL_SRC1_RGB GL_SOURCE1_RGB
#define GL_SRC2_ALPHA GL_SOURCE2_ALPHA
#define GL_SRC2_RGB GL_SOURCE2_RGB
#define GL_BUFFER_SIZE 0x8764
#define GL_BUFFER_USAGE 0x8765
#define GL_QUERY_COUNTER_BITS 0x8864
#define GL_CURRENT_QUERY 0x8865
#define GL_QUERY_RESULT 0x8866
#define GL_QUERY_RESULT_AVAILABLE 0x8867
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_ARRAY_BUFFER_BINDING 0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING 0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING 0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING 0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING 0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING 0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING 0x889A
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING 0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING 0x889C
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING 0x889D
#define GL_WEIGHT_ARRAY_BUFFER_BINDING 0x889E
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING 0x889F
#define GL_READ_ONLY 0x88B8
#define GL_WRITE_ONLY 0x88B9
#define GL_READ_WRITE 0x88BA
#define GL_BUFFER_ACCESS 0x88BB
#define GL_BUFFER_MAPPED 0x88BC
#define GL_BUFFER_MAP_POINTER 0x88BD
#define GL_STREAM_DRAW 0x88E0
#define GL_STREAM_READ 0x88E1
#define GL_STREAM_COPY 0x88E2
#define GL_STATIC_DRAW 0x88E4
#define GL_STATIC_READ 0x88E5
#define GL_STATIC_COPY 0x88E6
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_DYNAMIC_READ 0x88E9
#define GL_DYNAMIC_COPY 0x88EA
#define GL_SAMPLES_PASSED 0x8914

#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT 0x00000001
#define GL_ELEMENT_ARRAY_BARRIER_BIT 0x00000002
#define GL_UNIFORM_BARRIER_BIT 0x00000004
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x00000008
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#define GL_COMMAND_BARRIER_BIT 0x00000040
#define GL_PIXEL_BUFFER_BARRIER_BIT 0x00000080
#define GL_TEXTURE_UPDATE_BARRIER_BIT 0x00000100
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GL_FRAMEBUFFER_BARRIER_BIT 0x00000400
#define GL_TRANSFORM_FEEDBACK_BARRIER_BIT 0x00000800
#define GL_ATOMIC_COUNTER_BARRIER_BIT 0x00001000
#define GL_MAX_IMAGE_UNITS 0x8F38
#define GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS 0x8F39
#define GL_IMAGE_BINDING_NAME 0x8F3A
#define GL_IMAGE_BINDING_LEVEL 0x8F3B
#define GL_IMAGE_BINDING_LAYERED 0x8F3C
#define GL_IMAGE_BINDING_LAYER 0x8F3D
#define GL_IMAGE_BINDING_ACCESS 0x8F3E
#define GL_IMAGE_1D 0x904C
#define GL_IMAGE_2D 0x904D
#define GL_IMAGE_3D 0x904E
#define GL_IMAGE_2D_RECT 0x904F
#define GL_IMAGE_CUBE 0x9050
#define GL_IMAGE_BUFFER 0x9051
#define GL_IMAGE_1D_ARRAY 0x9052
#define GL_IMAGE_2D_ARRAY 0x9053
#define GL_IMAGE_CUBE_MAP_ARRAY 0x9054
#define GL_IMAGE_2D_MULTISAMPLE 0x9055
#define GL_IMAGE_2D_MULTISAMPLE_ARRAY 0x9056
#define GL_INT_IMAGE_1D 0x9057
#define GL_INT_IMAGE_2D 0x9058
#define GL_INT_IMAGE_3D 0x9059
#define GL_INT_IMAGE_2D_RECT 0x905A
#define GL_INT_IMAGE_CUBE 0x905B
#define GL_INT_IMAGE_BUFFER 0x905C
#define GL_INT_IMAGE_1D_ARRAY 0x905D
#define GL_INT_IMAGE_2D_ARRAY 0x905E
#define GL_INT_IMAGE_CUBE_MAP_ARRAY 0x905F
#define GL_INT_IMAGE_2D_MULTISAMPLE 0x9060
#define GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY 0x9061
#define GL_UNSIGNED_INT_IMAGE_1D 0x9062
#define GL_UNSIGNED_INT_IMAGE_2D 0x9063
#define GL_UNSIGNED_INT_IMAGE_3D 0x9064
#define GL_UNSIGNED_INT_IMAGE_2D_RECT 0x9065
#define GL_UNSIGNED_INT_IMAGE_CUBE 0x9066
#define GL_UNSIGNED_INT_IMAGE_BUFFER 0x9067
#define GL_UNSIGNED_INT_IMAGE_1D_ARRAY 0x9068
#define GL_UNSIGNED_INT_IMAGE_2D_ARRAY 0x9069
#define GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY 0x906A
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE 0x906B
#define GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY 0x906C
#define GL_MAX_IMAGE_SAMPLES 0x906D
#define GL_IMAGE_BINDING_FORMAT 0x906E
#define GL_IMAGE_FORMAT_COMPATIBILITY_TYPE 0x90C7
#define GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE 0x90C8
#define GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS 0x90C9
#define GL_MAX_VERTEX_IMAGE_UNIFORMS 0x90CA
#define GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS 0x90CB
#define GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS 0x90CC
#define GL_MAX_GEOMETRY_IMAGE_UNIFORMS 0x90CD
#define GL_MAX_FRAGMENT_IMAGE_UNIFORMS 0x90CE
#define GL_MAX_COMBINED_IMAGE_UNIFORMS 0x90CF
#define GL_ALL_BARRIER_BITS 0xFFFFFFFF
	
#define GL_COMPUTE_SHADER_BIT 0x00000020
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE 0x8262
#define GL_MAX_COMPUTE_UNIFORM_COMPONENTS 0x8263
#define GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS 0x8264
#define GL_MAX_COMPUTE_ATOMIC_COUNTERS 0x8265
#define GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS 0x8266
#define GL_COMPUTE_WORK_GROUP_SIZE 0x8267
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#define GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER 0x90EC
#define GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER 0x90ED
#define GL_DISPATCH_INDIRECT_BUFFER 0x90EE
#define GL_DISPATCH_INDIRECT_BUFFER_BINDING 0x90EF
#define GL_COMPUTE_SHADER 0x91B9
#define GL_MAX_COMPUTE_UNIFORM_BLOCKS 0x91BB
#define GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS 0x91BC
#define GL_MAX_COMPUTE_IMAGE_UNIFORMS 0x91BD
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT 0x91BE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE 0x91BF
	
#define GL_COMPRESSED_RED 0x8225
#define GL_COMPRESSED_RG 0x8226
#define GL_RG 0x8227
#define GL_RG_INTEGER 0x8228
#define GL_R8 0x8229
#define GL_R16 0x822A
#define GL_RG8 0x822B
#define GL_RG16 0x822C
#define GL_R16F 0x822D
#define GL_R32F 0x822E
#define GL_RG16F 0x822F
#define GL_RG32F 0x8230
#define GL_R8I 0x8231
#define GL_R8UI 0x8232
#define GL_R16I 0x8233
#define GL_R16UI 0x8234
#define GL_R32I 0x8235
#define GL_R32UI 0x8236
#define GL_RG8I 0x8237
#define GL_RG8UI 0x8238
#define GL_RG16I 0x8239
#define GL_RG16UI 0x823A
#define GL_RG32I 0x823B
#define GL_RG32UI 0x823C

typedef void		(__stdcall *t_glBlendEquation)(unsigned mode);
typedef void		(__stdcall *t_glGenBuffers)(int n, unsigned *buffers);
typedef void		(__stdcall *t_glBindBuffer)(unsigned target, unsigned buffer);
typedef void		(__stdcall *t_glBufferData)(unsigned target, int size, const void *data, unsigned usage);
typedef void		(__stdcall *t_glBufferSubData)(unsigned target, int offset, int size, const void *data);
typedef void		(__stdcall *t_glEnableVertexAttribArray)(unsigned index);
typedef void		(__stdcall *t_glVertexAttribPointer)(unsigned index, int size, unsigned type, unsigned char normalized, int stride, const void *pointer);
typedef void		(__stdcall *t_glDisableVertexAttribArray)(unsigned index);
typedef unsigned	(__stdcall *t_glCreateShader)(unsigned shaderType);
typedef void		(__stdcall *t_glShaderSource)(unsigned shader, int count, const char **string, const int *length);
typedef void		(__stdcall *t_glCompileShader)(unsigned shader);
typedef void		(__stdcall *t_glGetShaderiv)(unsigned shader, unsigned pname, int *params);
typedef void		(__stdcall *t_glGetShaderInfoLog)(unsigned shader, int maxLength, int *length, char *infoLog);
typedef unsigned	(__stdcall *t_glCreateProgram)();
typedef void		(__stdcall *t_glAttachShader)(unsigned program, unsigned shader);
typedef void		(__stdcall *t_glLinkProgram)(unsigned program);
typedef void		(__stdcall *t_glGetProgramiv)(unsigned program, unsigned pname, int *params);
typedef void		(__stdcall *t_glGetProgramInfoLog)(unsigned program, int maxLength, int *length, char *infoLog);
typedef void		(__stdcall *t_glDetachShader)(unsigned program, unsigned shader);
typedef void		(__stdcall *t_glDeleteShader)(unsigned shader);
typedef void		(__stdcall *t_glUseProgram)(unsigned program);
typedef int			(__stdcall *t_glGetAttribLocation)(unsigned program, const char *name);
typedef void		(__stdcall *t_glDeleteProgram)(unsigned program);
typedef void		(__stdcall *t_glDeleteBuffers)(int n, const unsigned *buffers);
typedef int			(__stdcall *t_glGetUniformLocation)(unsigned program, const char *name);
typedef void		(__stdcall *t_glUniformMatrix3fv)(int location, int count, unsigned char transpose, const float *value);
typedef void		(__stdcall *t_glUniformMatrix4fv)(int location, int count, unsigned char transpose, const float *value);
typedef void		(__stdcall *t_glGetBufferParameteriv)(unsigned target, unsigned value, int *data);
typedef void		(__stdcall *t_glActiveTexture)(unsigned texture);
typedef void		(__stdcall *t_glUniform1i)(int location, int v0);
typedef void		(__stdcall *t_glUniform2i)(int location, int v0, int v1);
typedef void		(__stdcall *t_glUniform1f)(int location, float v0);
typedef void		(__stdcall *t_glUniform2f)(int location, float v0, float v1);
typedef void		(__stdcall *t_glUniform3f)(int location, float v0, float v1, float v2);
typedef void		(__stdcall *t_glUniform3fv)(int location, int count, const float *value);
typedef void		(__stdcall *t_glUniform4f)(int location, float v0, float v1, float v2, float v3);
typedef void		(__stdcall *t_glUniform4fv)(int location, int count, float *value);
typedef void*		(__stdcall *t_glMapBuffer)(unsigned target, unsigned access);
typedef unsigned char(__stdcall *t_glUnmapBuffer)(unsigned target);

//OpenGL 3.0
typedef void		(__stdcall *t_glGenVertexArrays)(int n, unsigned *arrays);
typedef void		(__stdcall *t_glDeleteVertexArrays)(int n, unsigned *arrays);
typedef void		(__stdcall *t_glBindVertexArray)(unsigned arr);

//OpenGL 4.2
typedef void		(__stdcall *t_glTexStorage2D)(unsigned target, int levels, unsigned internalformat, int width, int height);
typedef void		(__stdcall *t_glBindImageTexture)(unsigned unit, unsigned texture, int level, unsigned char layered, int layer, unsigned access, unsigned format);
typedef void		(__stdcall *t_glMemoryBarrier)(unsigned barriers);

//OpenGL 4.3
typedef void		(__stdcall *t_glDispatchCompute)(unsigned num_groups_x, unsigned num_groups_y, unsigned num_groups_z);

#define		GLFUNCLIST\
	GLFUNC(glBlendEquation)\
	GLFUNC(glGenBuffers)\
	GLFUNC(glBindBuffer)\
	GLFUNC(glBufferData)\
	GLFUNC(glBufferSubData)\
	GLFUNC(glEnableVertexAttribArray)\
	GLFUNC(glVertexAttribPointer)\
	GLFUNC(glDisableVertexAttribArray)\
	GLFUNC(glCreateShader)\
	GLFUNC(glShaderSource)\
	GLFUNC(glCompileShader)\
	GLFUNC(glGetShaderiv)\
	GLFUNC(glGetShaderInfoLog)\
	GLFUNC(glCreateProgram)\
	GLFUNC(glAttachShader)\
	GLFUNC(glLinkProgram)\
	GLFUNC(glGetProgramiv)\
	GLFUNC(glGetProgramInfoLog)\
	GLFUNC(glDetachShader)\
	GLFUNC(glDeleteShader)\
	GLFUNC(glUseProgram)\
	GLFUNC(glGetAttribLocation)\
	GLFUNC(glDeleteProgram)\
	GLFUNC(glDeleteBuffers)\
	GLFUNC(glGetUniformLocation)\
	GLFUNC(glUniformMatrix3fv)\
	GLFUNC(glUniformMatrix4fv)\
	GLFUNC(glGetBufferParameteriv)\
	GLFUNC(glActiveTexture)\
	GLFUNC(glUniform1i)\
	GLFUNC(glUniform2i)\
	GLFUNC(glUniform1f)\
	GLFUNC(glUniform2f)\
	GLFUNC(glUniform3f)\
	GLFUNC(glUniform3fv)\
	GLFUNC(glUniform4f)\
	GLFUNC(glUniform4fv)\
	GLFUNC(glMapBuffer)\
	GLFUNC(glUnmapBuffer)\
	GLFUNC(glGenVertexArrays)\
	GLFUNC(glDeleteVertexArrays)\
	GLFUNC(glBindVertexArray)\
	GLFUNC(glTexStorage2D)\
	GLFUNC(glBindImageTexture)\
	GLFUNC(glMemoryBarrier)\
	GLFUNC(glDispatchCompute)

#define GLFUNC(X)	extern t_##X X;
GLFUNCLIST
#undef	GLFUNC

#endif//_MSC_VER

#define GL_CHECK(E)	(!(E=glGetError())||log_error(file, __LINE__, 1, "GL %d: %s", E, glerr2str(E)))
#define GL_ERROR(E)	(E=glGetError(), log_error(file, __LINE__, 1, "GL %d: %s", E, glerr2str(E)))
//void gl_check(const char *file, int line);
//void gl_error(const char *file, int line);
void init_gl();

#ifdef __cplusplus
}
#endif
#endif//INC_GLML_H
