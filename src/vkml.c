#include"acme_stdio.h"
#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include<Windows.h>
#elif defined __linux__
#include<dlfcn.h>
#endif
#include<stdlib.h>
#include"vkml.h"
#include"error.h"
#include"util.h"
#include"buffer.h"
#include"file.h"
static const char file[]=__FILE__;

#define VKFUNC(FUNCNAME)	PFN_##FUNCNAME FUNCNAME=0;
VKFUNCLIST1
VKFUNCLIST2
VKFUNCLIST3
#undef	VKFUNC

static int vk_loaded=0;
static void *hVulkan=0;
VkInstance instance=0;
VkDevice device=0;
VkQueue queue=0;

static const char *layernames[]=
{
	"VK_LAYER_KHRONOS_validation",
};
static const char *enabledInstExt[]=
{
	VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
};

const char* vk_err2str(int err)
{
	const char *a=0;
	switch(err)
	{
#define CASE(X)	case X:a=#X;break;
	CASE(VK_SUCCESS)
	CASE(VK_NOT_READY)
	CASE(VK_TIMEOUT)
	CASE(VK_EVENT_SET)
	CASE(VK_EVENT_RESET)
	CASE(VK_INCOMPLETE)
	CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
	CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
	CASE(VK_ERROR_INITIALIZATION_FAILED)
	CASE(VK_ERROR_DEVICE_LOST)
	CASE(VK_ERROR_MEMORY_MAP_FAILED)
	CASE(VK_ERROR_LAYER_NOT_PRESENT)
	CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
	CASE(VK_ERROR_FEATURE_NOT_PRESENT)
	CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
	CASE(VK_ERROR_TOO_MANY_OBJECTS)
	CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
	CASE(VK_ERROR_FRAGMENTED_POOL)
	CASE(VK_ERROR_UNKNOWN)
	CASE(VK_ERROR_OUT_OF_POOL_MEMORY)
	CASE(VK_ERROR_INVALID_EXTERNAL_HANDLE)
	CASE(VK_ERROR_FRAGMENTATION)
	CASE(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)
	CASE(VK_ERROR_SURFACE_LOST_KHR)
	CASE(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)
	CASE(VK_SUBOPTIMAL_KHR)
	CASE(VK_ERROR_OUT_OF_DATE_KHR)
	CASE(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR)
	CASE(VK_ERROR_VALIDATION_FAILED_EXT)
	CASE(VK_ERROR_INVALID_SHADER_NV)
	CASE(VK_ERROR_INCOMPATIBLE_VERSION_KHR)
	CASE(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT)
	CASE(VK_ERROR_NOT_PERMITTED_EXT)
	CASE(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
	CASE(VK_THREAD_IDLE_KHR)
	CASE(VK_THREAD_DONE_KHR)
	CASE(VK_OPERATION_DEFERRED_KHR)
	CASE(VK_OPERATION_NOT_DEFERRED_KHR)
	CASE(VK_PIPELINE_COMPILE_REQUIRED_EXT)
	CASE(VK_RESULT_MAX_ENUM)
#undef	CASE
	}
	if(!a)
		a="UNKNOWN ERROR";
	return a;
}
void finish_vk()
{
	vkDestroyDevice(device, 0);
	vkDestroyInstance(instance, 0);
}
int findProperties(const VkPhysicalDeviceMemoryProperties *memProps, unsigned memTypeBitsReq, VkMemoryPropertyFlags reqProps)
{
	//https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
	for (unsigned idx=0;idx<memProps->memoryTypeCount;++idx)
	{
		if((memTypeBitsReq&(1<<idx))&&(memProps->memoryTypes[idx].propertyFlags&reqProps)==reqProps)//is required memory type and has required properties
			return idx;
	}
	return -1;//failed to find memory type
}

#if 1
typedef struct ComponentTypeInfoStruct
{
    const char *typeName;
    unsigned bits;
} ComponentTypeInfo;
typedef enum TestTypeEnum
{
    TT_SHARED,
    TT_TILED,
    TT_COUNT,
} TestType;
ComponentTypeInfo componentTypeInfo[]=
{
	{"float16_t",	16},
	{"float32_t",	32},
	{"float64_t",	64},
	{"int8_t",		 8},
	{"int16_t",		16},
	{"int32_t",		32},
	{"int64_t",		64},
	{"uint8_t",		 8},
	{"uint16_t",	16},
	{"uint32_t",	32},
	{"uint64_t",	64},
};
typedef struct TestCaseStruct
{
    TestType testType;
    VkComponentTypeNV inputType, outputType;

    unsigned M, N, K;// MxNxK is the size of the full matrix multiply
    unsigned lM, lN, lK;// Each cooperative matrix multiply is lMxlNxlK
    unsigned TILE_M, TILE_N, TILE_K;// size of workgroup tile in destination matrix

    int BColMajor;
    unsigned ARowLen, ANumRows;
    unsigned BRowLen, BNumRows;
} TestCase;
typedef struct MatrixDescStruct
{
    struct
    {
        unsigned rows, cols;
    } dims;
    VkComponentTypeNV dataType;
    size_t elementSize;
    VkDeviceSize bufferSize;
    unsigned totalElements;

    //Create a host- and device-local buffer for each input and output.
    //Descriptors point at the device buffers.
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkDeviceMemory deviceMemory;
    void *ptr;
} MatrixDesc;
int isFloatType(MatrixDesc const *m)
{
    switch (m->dataType)
    {
    case VK_COMPONENT_TYPE_FLOAT16_NV:
    case VK_COMPONENT_TYPE_FLOAT32_NV:
    case VK_COMPONENT_TYPE_FLOAT64_NV:
        return 1;
    default:
        return 0;
    }
}
void setDataFloat(MatrixDesc *m, unsigned i, float value)
{
    if (m->dataType == VK_COMPONENT_TYPE_FLOAT32_NV)
        ((float *)m->ptr)[i]=value;
    else
    {
        unsigned asInt=*(unsigned*)&value;
        int sign=(asInt&0x80000000)>>31;
        int exp=((asInt&0x7f800000)>>23)-127;
        int mantissa=(asInt & 0x7FFFFF);

        sign=sign<<15;
        exp=(exp+15)<<10;
        mantissa=mantissa>>(23-10);

        if(asInt)
            asInt=sign|exp|mantissa;

        ((uint16_t*)m->ptr)[i]=asInt;
    }
}
float getDataFloat(MatrixDesc const *m, unsigned i)
{
    if (m->dataType == VK_COMPONENT_TYPE_FLOAT32_NV)
        return ((float*)m->ptr)[i];
    else
    {
        unsigned asInt=((uint16_t*)m->ptr)[i];
        int sign=(asInt&0x8000)>>15;
        int exp=((asInt&0x7c00)>>10)-15;
        int mantissa=(asInt&0x3FF);

        sign = sign << 31;
        exp = (exp + 127) << 23;
        mantissa = mantissa << (23 - 10);

        if(asInt)
            asInt=sign|exp|mantissa;

        return *(float*)&asInt;
    }
}
float getDataFloat_c(MatrixDesc const *mat, int ky, int kx, int colMajor)
{
    return getDataFloat(mat, colMajor?kx*mat->dims.rows+ky:ky*mat->dims.cols+kx);
}
void setDataInt(MatrixDesc *m, unsigned i, unsigned value)
{
    ASSERT_MSG(componentTypeInfo[m->dataType].bits==8||componentTypeInfo[m->dataType].bits==32, "Invalid number of bits");
    switch(m->dataType)
	{
    case VK_COMPONENT_TYPE_UINT8_NV:    ((uint8_t	*)m->ptr)[i]=(uint8_t	)value; break;
    case VK_COMPONENT_TYPE_UINT32_NV:   ((unsigned	*)m->ptr)[i]=(unsigned	)value; break;
    case VK_COMPONENT_TYPE_SINT8_NV:    ((int8_t	*)m->ptr)[i]=(int8_t	)value; break;
    case VK_COMPONENT_TYPE_SINT32_NV:   ((int		*)m->ptr)[i]=(int		)value; break;
    default: ASSERT_MSG(0, "Invalid data type");//fallthrough
    }
}
unsigned getDataInt(MatrixDesc const *mat, unsigned i)
{
    ASSERT_MSG(componentTypeInfo[mat->dataType].bits==8||componentTypeInfo[mat->dataType].bits==32, "Invalid number of bits");
	switch(mat->dataType)
	{
	case VK_COMPONENT_TYPE_UINT8_NV:	return ((uint8_t	*)mat->ptr)[i];
	case VK_COMPONENT_TYPE_UINT32_NV:	return ((unsigned	*)mat->ptr)[i];
	case VK_COMPONENT_TYPE_SINT8_NV:	return ((int8_t		*)mat->ptr)[i];
	case VK_COMPONENT_TYPE_SINT32_NV:	return ((int		*)mat->ptr)[i];
	default: ASSERT_MSG(0, "Invalid data type");//fallthrough
	}
	return 0;
}
unsigned getDataInt_c(MatrixDesc const *mat, int ky, int kx, int colMajor)
{
    return getDataInt(mat, colMajor?kx*mat->dims.rows+ky:ky*mat->dims.cols+kx);
}
void createMatrixDesc(VkDevice device, VkPhysicalDeviceMemoryProperties *memoryProperties, MatrixDesc *m, VkComponentTypeNV dt, int rows, int cols)
{
	VkResult error;

	m->dims.rows=rows;
	m->dims.cols=cols;
	m->dataType=dt;
	m->elementSize=componentTypeInfo[m->dataType].bits/8;
	m->totalElements=m->dims.cols*m->dims.rows;
	m->bufferSize=m->totalElements*m->elementSize;

	VkBufferCreateInfo bufferCreateInfo=
	{
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0,
		m->bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
		VK_SHARING_MODE_EXCLUSIVE,
		0u,
		NULL,
	};
	error=vkCreateBuffer(device, &bufferCreateInfo, NULL, &m->hostBuffer); VKCHECK(error);
	error=vkCreateBuffer(device, &bufferCreateInfo, NULL, &m->deviceBuffer); VKCHECK(error);

	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(device, m->hostBuffer, &memReqs);
	int32_t hostIndex=findProperties(memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
	int32_t deviceIndex=findProperties(memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkMemoryAllocateInfo memAllocateInfo=
	{
		VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		NULL,
		memReqs.size,
		(uint32_t)hostIndex,
	};
	error=vkAllocateMemory(device, &memAllocateInfo, 0, &m->hostMemory); VKCHECK(error);

	memAllocateInfo.memoryTypeIndex=deviceIndex;
	error=vkAllocateMemory(device, &memAllocateInfo, 0, &m->deviceMemory); VKCHECK(error);

	error=vkBindBufferMemory(device, m->hostBuffer, m->hostMemory, 0); VKCHECK(error);
	error=vkBindBufferMemory(device, m->deviceBuffer, m->deviceMemory, 0); VKCHECK(error);
	error=vkMapMemory(device, m->hostMemory, 0, m->bufferSize, 0, &m->ptr); VKCHECK(error);
}
void destroyMatrixDesc(VkDevice device, MatrixDesc *m)//destroy storage for a matrix
{
    vkDestroyBuffer(device, m->hostBuffer, 0);
    vkDestroyBuffer(device, m->deviceBuffer, 0);
    vkFreeMemory(device, m->hostMemory, 0);
    vkFreeMemory(device, m->deviceMemory, 0);
}
#endif
void init_vk()
{
	VkResult error;
	if(!vk_loaded)
	{
#ifdef _WIN32
		hVulkan=LoadLibraryA("vulkan-1.dll");
#elif defined __linux__
		hVulkan=dlopen("libvulkan.so", RTLD_NOW);
#endif
		ASSERT_MSG(hVulkan, "Failed to load Vulkan DLL");

#ifdef _WIN32
#define VKFUNC(X)		(FARPROC)X=GetProcAddress(hVulkan, #X); ASSERT_MSG(X, "GetProcAddress(" #X ") returned NULL");
#elif defined __linux__
#define VKFUNC(X)		(void(*)(void))X=(void(*)(void))dlsym(hVulkan, #X); ASSERT_MSG(X, "Could not get " #X);
#endif
		VKFUNCLIST1
#undef	VKFUNC
			
#define VKFUNC(X)		(PFN_vkVoidFunction)X=vkGetInstanceProcAddr(instance, #X); ASSERT_MSG(X, "vkGetInstanceProcAddr(" #X ") returned NULL");
		VKFUNCLIST2
		
		VkApplicationInfo appinfo=
		{
			VK_STRUCTURE_TYPE_APPLICATION_INFO, 0,
			"VKML", 1,//application
			0, 0,//engine
			VK_API_VERSION_1_2,
		};
		VkInstanceCreateInfo instinfo=
		{
			VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0,
			0, &appinfo,

			//COUNTOF(layernames), layernames,//layers
			0, 0,

			COUNTOF(enabledInstExt), enabledInstExt,//extensions
		};
		VkResult error=vkCreateInstance(&instinfo, 0, &instance);	VKCHECK(error);
		ASSERT_MSG(instance, "Failed to create Vulkan instance");

		VKFUNCLIST3
#undef	VKFUNC

		vk_loaded=1;
	}

	//select a physical device
#if 1
	unsigned npdevices=0;
	VkPhysicalDevice *pdevices=0;
	error=vkEnumeratePhysicalDevices(instance, &npdevices, 0);	VKCHECK(error);
	ASSERT_MSG(npdevices, "No Vulkan physical devices");
	pdevices=(VkPhysicalDevice*)malloc(npdevices*sizeof(VkPhysicalDevice));
	error=vkEnumeratePhysicalDevices(instance, &npdevices, pdevices);	VKCHECK(error);
	
	int idx=-1;
	for(unsigned k=0;k<npdevices;++k)
	{
		unsigned numExtensions=0;
		ArrayHandle extensions;

		error=vkEnumerateDeviceExtensionProperties(pdevices[k], 0, &numExtensions, 0);	VKCHECK(error);
		ARRAY_ALLOC(VkExtensionProperties, extensions, 0, numExtensions, 0, 0);
		error=vkEnumerateDeviceExtensionProperties(pdevices[k], 0, &numExtensions, (VkExtensionProperties*)extensions->data);	VKCHECK(error);

		//printf("Extensions:\n");
		//for(unsigned k2=0;k2<numExtensions;++k2)//
		//{
		//	VkExtensionProperties *ext=(VkExtensionProperties*)array_at(&extensions, k2);
		//	printf("\t%s\t%d\n", ext->extensionName, ext->specVersion);
		//}

		for(unsigned k2=0;k2<numExtensions;++k2)
		{
			VkExtensionProperties *ext=(VkExtensionProperties*)array_at(&extensions, k2);
			if(!strcmp(ext->extensionName, VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME))
			{
				idx=k;
				break;
			}
		}
		array_free(&extensions);
		if(idx!=-1)
			break;
	}
	ASSERT_MSG(idx!=-1, "VK_NV_cooperative_matrix is not supported");
	VkPhysicalDevice pdevice=pdevices[idx];
	free(pdevices);
	pdevices=0;

	VkPhysicalDeviceProperties pdevprop={0};
	vkGetPhysicalDeviceProperties(pdevice, &pdevprop);
	printf("%s  API: %d.%d.%d\n", pdevprop.deviceName, VK_VERSION_MAJOR(pdevprop.apiVersion), VK_VERSION_MINOR(pdevprop.apiVersion), VK_VERSION_PATCH(pdevprop.apiVersion));
#endif

	//create logical device			https://github.com/jeffbolznv/vk_cooperative_matrix_perf
#if 1
	//select compute queue
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(pdevice, &memoryProperties);

	unsigned nqfams=0;
	ArrayHandle qfams=0;
	vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &nqfams, 0);
	ARRAY_ALLOC(VkQueueFamilyProperties, qfams, 0, nqfams, 0, 0);
	vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &nqfams, (VkQueueFamilyProperties*)qfams->data);
	idx=-1;
	for(unsigned k=0;k<nqfams;++k)
	{
		VkQueueFamilyProperties *qfam=(VkQueueFamilyProperties*)array_at(&qfams, k);
		if(qfam->queueFlags&VK_QUEUE_COMPUTE_BIT)
		{
			idx=k;
			break;
		}
	}
	array_free(&qfams);
	ASSERT_MSG(idx!=-1, "Couldn't find compute queue");
	int queueFamilyIdx=idx;
	float queuePriority=1.f;
	VkDeviceQueueCreateInfo deviceQueueCreateInfo=
	{
		VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0,
		queueFamilyIdx,
		1,
		&queuePriority,
	};

	//Query the list of supported cooperative matrix multiply sizes/types.
	unsigned nCoopMatProps=0;
	ArrayHandle coopMatProps=0;

#define	VKFUNC(X)	PFN_##X X=(PFN_##X)vkGetInstanceProcAddr(instance, #X);
	VKFUNC(vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)
#undef	VKFUNC

	error=vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(pdevice, &nCoopMatProps, 0);	VKCHECK(error);
	ARRAY_ALLOC(VkCooperativeMatrixPropertiesNV, coopMatProps, 0, nCoopMatProps, 0, 0);
	for (int k=0;k<(int)nCoopMatProps;++k)
	{
		VkCooperativeMatrixPropertiesNV *prop=(VkCooperativeMatrixPropertiesNV*)array_at(&coopMatProps, k);
		prop->sType=VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV;
		prop->pNext=0;
	}
	error=vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(pdevice, &nCoopMatProps, (VkCooperativeMatrixPropertiesNV*)coopMatProps->data); VKCHECK(error);

	//deviceCreateInfo -> float16Features -> bufferDeviceAddressFeatures -> coopMatFeatures -> 0
	VkPhysicalDeviceCooperativeMatrixFeaturesNV coopMatFeatures=
	{
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV, 0,
		VK_TRUE,	//cooperativeMatrix
		VK_FALSE,	//cooperativeMatrixRobustBufferAccess
	};
	VkPhysicalDeviceBufferAddressFeaturesEXT bufferDeviceAddressFeatures=
	{
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT, &coopMatFeatures,
		VK_TRUE,	//bufferDeviceAddress
		VK_FALSE,	//bufferDeviceAddressCaptureReplay
		VK_FALSE,	//bufferDeviceAddressMultiDevice
	};
	VkPhysicalDeviceFloat16Int8FeaturesKHR float16Features=
	{
		VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR, &bufferDeviceAddressFeatures,
		VK_TRUE,	//shaderFloat16
		VK_FALSE,	//shaderInt8
	};
	const char *enabledDeviceExtensions[]=
	{
		VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME,
		VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
		VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
	};
	VkDeviceCreateInfo deviceCreateInfo=
	{
		VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &float16Features, 0,
		1, &deviceQueueCreateInfo,	//queue(s)
		0, 0,						//layer(s)
		COUNTOF(enabledDeviceExtensions), enabledDeviceExtensions,//extension(s)
		0,							//feature(s)
	};
	error=vkCreateDevice(pdevice, &deviceCreateInfo, 0, &device);	VKCHECK(error);
	vkGetDeviceQueue(device, idx, 0, &queue);
#endif

	//
#if 1
	//The shaders use one UBO to pass in all the buffer addresses
	VkDescriptorSetLayoutBinding layoutBinding=
	{
		0,
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
		VK_SHADER_STAGE_COMPUTE_BIT,
		0,
	};
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo=
	{
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0,
		1,
		&layoutBinding,
	};
	VkDescriptorSetLayout descriptorSetLayout;
	error=vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0, &descriptorSetLayout); VKCHECK(error);

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo=
	{
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0,
		1, &descriptorSetLayout,
		0, 0,
	};
	VkPipelineLayout pipelineLayout;
	error=vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, &pipelineLayout); VKCHECK(error);

	VkDescriptorPoolSize poolSizes[1]=
	{
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}
	};
	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo=
	{
		VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0,
		VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		1,
		COUNTOF(poolSizes), poolSizes,
	};
	VkDescriptorPool descriptorPool;
	error=vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, 0, &descriptorPool); VKCHECK(error);

	VkDescriptorSetAllocateInfo setAllocateInfo=
	{
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0,
		descriptorPool,
		1, &descriptorSetLayout,
	};
	VkDescriptorSet descriptorSet;
	error=vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet); VKCHECK(error);

	VkCommandPoolCreateInfo commandPoolCreateInfo=
	{
		VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		queueFamilyIdx,
	};
	VkCommandPool commandPool;
	error = vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool); VKCHECK(error);
	
	//The command buffers,
	//	one for initializing buffers,
	//	one for compute,
	//	one for reading back the results.
	//This lets us time the compute work more precisely.
	VkCommandBufferAllocateInfo commandBufferAllocateInfo=
	{
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0,
		commandPool,
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		3,
	};

	VkCommandBuffer commandBuffers[3];
	error=vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers); VKCHECK(error);
	static const char *scopeString[]=
	{
		"invalid",
		"device",
		"workgroup",
		"subgroup",
		"invalid",
		"queuefamily",
	};
#endif

	//main loop
#if 1
	for(int tt=0;tt<TT_COUNT;++tt)//Loop over all shader types and all cooperative matrix properties.
	{
		for (int i=0;i<(int)nCoopMatProps;++i)
		{
			VkCooperativeMatrixPropertiesNV *cooperativeMatrixProps=(VkCooperativeMatrixPropertiesNV*)array_at(&coopMatProps, i);

			if (cooperativeMatrixProps->DType!=VK_COMPONENT_TYPE_FLOAT16_NV &&
				cooperativeMatrixProps->DType!=VK_COMPONENT_TYPE_FLOAT32_NV &&
				cooperativeMatrixProps->AType!=VK_COMPONENT_TYPE_UINT8_NV &&
				cooperativeMatrixProps->AType!=VK_COMPONENT_TYPE_SINT8_NV)
				continue;
			ArrayHandle filename=0;
			STR_ALLOC(filename, 0);

			//std::string fileName;
			static const char *prefixes[]=
			{
				"shader_shmem",
				"shader_tiled",
			};
			switch(tt)
			{
			case TT_SHARED:
				STR_APPEND(filename, prefixes[0], strlen(prefixes[0]), 1);
				//fileName=std::string("shaders/shmem");
				break;
			case TT_TILED:
				STR_APPEND(filename, prefixes[1], strlen(prefixes[1]), 1);
				//fileName=std::string("shaders/tiled");
				break;
			default:
				ASSERT_MSG(0, "Invalid main loop counter");
			}
			if(cooperativeMatrixProps->AType==VK_COMPONENT_TYPE_UINT8_NV)
				STR_APPEND(filename, "u8", 2, 1);
			else if(cooperativeMatrixProps->AType==VK_COMPONENT_TYPE_SINT8_NV)
				STR_APPEND(filename, "s8", 2, 1);
			else if(cooperativeMatrixProps->DType==VK_COMPONENT_TYPE_FLOAT16_NV)
				STR_APPEND(filename, "fp16", 4, 1);
			else
				STR_APPEND(filename, "fp32", 4, 1);
			STR_APPEND(filename, ".spv", 4, 1);
			//std::string suffix =
			//	cooperativeMatrixProps->AType == VK_COMPONENT_TYPE_UINT8_NV ? "u8" :
			//	cooperativeMatrixProps->AType == VK_COMPONENT_TYPE_SINT8_NV ? "s8" :
			//	cooperativeMatrixProps->DType == VK_COMPONENT_TYPE_FLOAT16_NV ? "fp16" : "fp32";
			//fileName = fileName + suffix + ".spv";

			printf("\nshader: %s\n", filename->data);

			//Load and create the shader module.
			ArrayHandle spirv=load_file(filename->data, 1, 0);
			array_free(&filename);
			//std::ifstream spirvfile(fileName.c_str(), std::ios::binary | std::ios::ate);
			//std::streampos spirvsize = spirvfile.tellg();
			//if ((int)spirvsize == -1) {
			//	printf("%s not found!\n", fileName.c_str());
			//	throw;
			//}
			//spirvfile.seekg(0, std::ios::beg);
			//
			//vector<char> spirv(spirvsize);
			//spirvfile.read(&spirv[0], spirvsize);

			VkShaderModuleCreateInfo shaderModuleCreateInfo=
			{
				VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				NULL,
				0,
				spirv->count,
				(const unsigned*)spirv->data,
			};

			VkShaderModule shaderModule;
			error=vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &shaderModule); VKCHECK(error);

			printf("\ncooperativeMatrixProps = %dx%dx%d   A = %s B = %s C = %s D = %s scope = %s\n",
					cooperativeMatrixProps->MSize,
					cooperativeMatrixProps->NSize,
					cooperativeMatrixProps->KSize,
					componentTypeInfo[cooperativeMatrixProps->AType].typeName,
					componentTypeInfo[cooperativeMatrixProps->BType].typeName,
					componentTypeInfo[cooperativeMatrixProps->CType].typeName,
					componentTypeInfo[cooperativeMatrixProps->DType].typeName,
					scopeString[cooperativeMatrixProps->scope]);

			// For performance, test a 4096x4096x4096 multiply. For correctness,
			// test 256x256x256 (because the CPU reference computation is so slow).
			int correctness=1;
			unsigned defaultDim = correctness ? 256 : 4096, defaultM = defaultDim, defaultN = defaultDim, defaultK = defaultDim;

			typedef struct
			{
				unsigned int maxTILE_M;
				unsigned int maxTILE_N;
				unsigned int granularityTILE_M;
				unsigned int granularityTILE_N;
			} SubTestParams;

			// TT_SHARED requires a multiple of 128x128 to satisfy the assumptions
			// of its SSBO->shared memory copy code.
			SubTestParams subTestParams[]=
			{
				{ 256, 256, 128, 128 }, // TT_SHARED
				{ 128, 128, cooperativeMatrixProps->MSize, cooperativeMatrixProps->NSize }, // TT_TILED
			};

			SubTestParams *params = &subTestParams[tt];

			for (unsigned int TILE_M_size = params->granularityTILE_M; TILE_M_size <= params->maxTILE_M; TILE_M_size += params->granularityTILE_M)
			{
				double maxPerfThisIter = 0;
				for (unsigned int TILE_N_size = params->granularityTILE_N; TILE_N_size <= params->maxTILE_N; TILE_N_size += params->granularityTILE_N)
				{
					for (unsigned int bcolmajor = 0; bcolmajor <= 1; ++bcolmajor)
					{

						int BColMajor = bcolmajor != 0;

						// B matrix must be wide enough to load via uvec4 addressing from shared memory
						if (!BColMajor && tt == TT_SHARED &&
							componentTypeInfo[cooperativeMatrixProps->BType].bits / 8 * cooperativeMatrixProps->NSize < 16) {
							continue;
						}

						TestCase testCase = {
							(TestType)tt, //TestType testType;
							cooperativeMatrixProps->AType, // VkComponentTypeNV inputType;
							cooperativeMatrixProps->DType, // VkComponentTypeNV outputType;

							// MxNxK is the size of the full matrix multiply
							defaultM, // uint32_t M;
							defaultN, // uint32_t N;
							defaultK, // uint32_t K;

							// Each cooperative matrix multiply is lMxlNxlK
							cooperativeMatrixProps->MSize, // uint32_t lM;
							cooperativeMatrixProps->NSize, // uint32_t lN;
							cooperativeMatrixProps->KSize, // uint32_t lK;

							// size of workgroup tile in destination matrix
							TILE_M_size, // uint32_t TILE_M;
							TILE_N_size, // uint32_t TILE_N;
							cooperativeMatrixProps->KSize, // uint32_t TILE_K;

							BColMajor, // bool BColMajor;
						};
						float alpha = 2.0f, beta = 3.0f;

						if (tt == TT_SHARED) {
							// These TILE_K sizes are what happens to perform better on current HW.
							if (componentTypeInfo[cooperativeMatrixProps->AType].bits == 8) {
								testCase.TILE_K = 64;
							} else if (cooperativeMatrixProps->DType == VK_COMPONENT_TYPE_FLOAT16_NV) {
								testCase.TILE_K = 32;
							} else {
								testCase.TILE_K = 16;
							}
							// This tile size is too slow and may TDR.
							if (componentTypeInfo[cooperativeMatrixProps->DType].bits == 32 &&
								testCase.TILE_M == 256 && testCase.TILE_N == 256) {
								continue;
							}
						}

						// For non-power of two tile sizes, round up the matrix size to
						// be an even multiple of the tile size.
						testCase.M = (testCase.M + testCase.TILE_M - 1) / testCase.TILE_M * testCase.TILE_M;
						testCase.N = (testCase.N + testCase.TILE_N - 1) / testCase.TILE_N * testCase.TILE_N;
						testCase.K = (testCase.K + testCase.TILE_K - 1) / testCase.TILE_K * testCase.TILE_K;

						testCase.ARowLen = testCase.TILE_K;
						testCase.ANumRows = testCase.TILE_M;
						testCase.BRowLen = BColMajor ? testCase.TILE_K : testCase.TILE_N;
						testCase.BNumRows = BColMajor ? testCase.TILE_N : testCase.TILE_K;

						enum {MAT_A = 0, MAT_B = 1, MAT_C = 2, MAT_D = 3, NUM_MATS = 4};

						MatrixDesc matrices[NUM_MATS];

						createMatrixDesc(device, &memoryProperties, matrices+MAT_A, cooperativeMatrixProps->AType, testCase.M, testCase.K);
						createMatrixDesc(device, &memoryProperties, matrices+MAT_B, cooperativeMatrixProps->AType, testCase.K, testCase.N);
						createMatrixDesc(device, &memoryProperties, matrices+MAT_C, cooperativeMatrixProps->DType, testCase.M, testCase.N);
						createMatrixDesc(device, &memoryProperties, matrices+MAT_D, cooperativeMatrixProps->DType, testCase.M, testCase.N);

						// Allocate buffer to hold device addresses for the four matrices
						VkBuffer paramBuffer;
						VkDeviceMemory paramMemory;
						void *paramPtr;

						VkBufferCreateInfo bufferCreateInfo = {
							VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0,
							0,
							4*sizeof(VkDeviceAddress),
							VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							VK_SHARING_MODE_EXCLUSIVE,
							0u,
							0,
						};

						error=vkCreateBuffer(device, &bufferCreateInfo, 0, &paramBuffer); VKCHECK(error);

						VkMemoryRequirements memReqs;
						vkGetBufferMemoryRequirements(device, paramBuffer, &memReqs);

						int hostIndex = findProperties(&memoryProperties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

						VkMemoryAllocateInfo memAllocateInfo = {
							VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0,
							memReqs.size,
							(uint32_t)hostIndex,
						};

						error=vkAllocateMemory(device, &memAllocateInfo, 0, &paramMemory); VKCHECK(error);
						error=vkBindBufferMemory(device, paramBuffer, paramMemory, 0); VKCHECK(error);
						error=vkMapMemory(device, paramMemory, 0, bufferCreateInfo.size, 0, &paramPtr); VKCHECK(error);

						PFN_vkGetBufferDeviceAddressEXT pfn_vkGetBufferDeviceAddressEXT =
							(PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressEXT");

						for (int i=0;i<NUM_MATS;++i)
						{
							MatrixDesc *m=matrices+i;

							VkBufferDeviceAddressInfoEXT info=
							{
								VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT, 0,
								0,
							};
							VkDeviceAddress *addrsInMemory = (VkDeviceAddress *)paramPtr;
							info.buffer = m->deviceBuffer;
							VkDeviceAddress addr = pfn_vkGetBufferDeviceAddressEXT(device, &info);
							addrsInMemory[i] = addr;
						}

						VkDescriptorBufferInfo bufferDescriptor;
						bufferDescriptor.buffer = paramBuffer;
						bufferDescriptor.offset = 0;
						bufferDescriptor.range = bufferCreateInfo.size;

						VkWriteDescriptorSet writeDescriptorset=
						{
							VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0,
							descriptorSet,
							0,	//dstBinding,
							0,	//dstArrayElement
							1,	//descriptorCount
							VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
							0,
							&bufferDescriptor,
							0,
						};

						vkUpdateDescriptorSets(device, 1, &writeDescriptorset, 0, 0);

						//Initialize input buffers to random values.
						//These are relatively small and have few mantissa bits set so we don't lose precision in fp16 mode when running the correctness test.
						//Initialize the output buffer to an obvious value.
						for(unsigned i=0;i<NUM_MATS;++i)
						{
							MatrixDesc *m=matrices+i;
							for(unsigned j=0;j<m->totalElements;++j)
							{
								if(isFloatType(m))
								{
									if(i==3)
										setDataFloat(m, j, 1234.0f);
									else
										setDataFloat(m, j, ((float)(rand()&0x3)-1.0f)/2.0f);
								}
								else
								{
									if(i==3)
										setDataInt(m, j, 1234);
									else
										setDataInt(m, j, (rand()&0xFF)-128);
								}
							}
						}

						//Specialize the shader with the matrix sizes, strides, and constants.
						const unsigned specData[]=
						{
							testCase.lM,
							testCase.lN,
							testCase.lK,
							testCase.TILE_M,
							testCase.TILE_N,
							testCase.TILE_K,
							testCase.K,
							testCase.K, // stride0
							testCase.BColMajor ? testCase.K : testCase.N, // stride1
							testCase.N, // stride2
							testCase.N, // stride3
							*(uint32_t *)&alpha,
							*(uint32_t *)&beta,
							testCase.BColMajor,
							testCase.ARowLen,
							testCase.ANumRows,
							testCase.BRowLen,
							testCase.BNumRows,
						};

#if 0
						for(int i=0;i<COUNTOF(specData);++i)
							printf("specdata[%d] = %d\n", i, specData[i]);
#endif

						VkSpecializationMapEntry entries[]=
						{
							{ 0, sizeof(uint32_t)* 0, sizeof(uint32_t)},
							{ 1, sizeof(uint32_t)* 1, sizeof(uint32_t)},
							{ 2, sizeof(uint32_t)* 2, sizeof(uint32_t)},
							{ 3, sizeof(uint32_t)* 3, sizeof(uint32_t)},
							{ 4, sizeof(uint32_t)* 4, sizeof(uint32_t)},
							{ 5, sizeof(uint32_t)* 5, sizeof(uint32_t)},
							{ 6, sizeof(uint32_t)* 6, sizeof(uint32_t)},
							{ 7, sizeof(uint32_t)* 7, sizeof(uint32_t)},
							{ 8, sizeof(uint32_t)* 8, sizeof(uint32_t)},
							{ 9, sizeof(uint32_t)* 9, sizeof(uint32_t)},
							{10, sizeof(uint32_t)*10, sizeof(uint32_t)},
							{11, sizeof(uint32_t)*11, sizeof(uint32_t)},
							{12, sizeof(uint32_t)*12, sizeof(uint32_t)},
							{13, sizeof(uint32_t)*13, sizeof(uint32_t)},
							{14, sizeof(uint32_t)*14, sizeof(uint32_t)},
							{15, sizeof(uint32_t)*15, sizeof(uint32_t)},
							{16, sizeof(uint32_t)*16, sizeof(uint32_t)},
							{17, sizeof(uint32_t)*17, sizeof(uint32_t)},
						};
						VkSpecializationInfo specInfo =
						{
							COUNTOF(specData),
							entries,
							sizeof(specData),
							specData,
						};
						VkComputePipelineCreateInfo pipelineCreateInfo=
						{
							VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
							{
								VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0,
								VK_SHADER_STAGE_COMPUTE_BIT,
								shaderModule,
								"main",
								&specInfo,
							},
							pipelineLayout,
							VK_NULL_HANDLE,
							0,
						};

						VkPipeline pipeline=0;
						error=vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, 0, &pipeline); VKCHECK(error);

						VkCommandBufferBeginInfo commandBufferBeginInfo=
						{
							VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, 0,
							0,
						};

						// Download input buffers to device memory.
						error=vkBeginCommandBuffer(commandBuffers[0], &commandBufferBeginInfo); VKCHECK(error);

						for (unsigned i=0;i<4;++i)
						{
							MatrixDesc *m = matrices+i;
							VkBufferCopy copy={0, 0, m->bufferSize};
							vkCmdCopyBuffer(commandBuffers[0], m->hostBuffer, m->deviceBuffer, 1, &copy);
						}

						error=vkEndCommandBuffer(commandBuffers[0]); VKCHECK(error);

						VkSubmitInfo submitInfo=
						{
							VK_STRUCTURE_TYPE_SUBMIT_INFO,0, 0,
							0, 0,
							1, &commandBuffers[0],//sic
							0, 0,
						};
						submitInfo.pCommandBuffers=&commandBuffers[0];
						error=vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); VKCHECK(error);
						error=vkQueueWaitIdle(queue); VKCHECK(error);

						// Run the shader.
						error=vkBeginCommandBuffer(commandBuffers[1], &commandBufferBeginInfo); VKCHECK(error);

						vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);
						vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

						unsigned repeatCount=correctness?1:10;

						for(unsigned i=0;i<repeatCount;++i)
							vkCmdDispatch(commandBuffers[1], testCase.N/testCase.TILE_N, testCase.M/testCase.TILE_M, 1);

						error=vkEndCommandBuffer(commandBuffers[1]); VKCHECK(error);

						if(!correctness)
						{
							//warmup submits, to get the clocks up before we run the timing
							submitInfo.pCommandBuffers=&commandBuffers[1];
							int warmupCount=tt==TT_SHARED?5:2;
							for(int i=0;i<warmupCount;++i)
							{
								error=vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); VKCHECK(error);
								error=vkQueueWaitIdle(queue); VKCHECK(error);
							}
						}

						//Time the submit/wait time for this command buffer.
						double beginTime=time_ms();
						submitInfo.pCommandBuffers=&commandBuffers[1];
						error=vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); VKCHECK(error);
						error=vkQueueWaitIdle(queue); VKCHECK(error);
						double endTime=time_ms();

						double flops=2.*testCase.M*testCase.N*testCase.K*repeatCount;
						double tflops=(double)flops/(double)((endTime-beginTime)/1000000.0)/(1000.0*1000.0*1000.0*1000.0);

						printf("TILE_M=%d TILE_N=%d, TILE_K=%d BColMajor=%d ", testCase.TILE_M, testCase.TILE_N, testCase.TILE_K, testCase.BColMajor);
						//if(!correctness)
							printf("  %f TFlops\n", tflops);
						//else
						//	printf("\n");

						// Upload the result from device memory.
						error=vkBeginCommandBuffer(commandBuffers[2], &commandBufferBeginInfo); VKCHECK(error);
						{
							MatrixDesc *m=matrices+MAT_D;
							VkBufferCopy copy={0, 0, m->bufferSize};
							vkCmdCopyBuffer(commandBuffers[2], m->deviceBuffer, m->hostBuffer, 1, &copy);
						}
						error=vkEndCommandBuffer(commandBuffers[2]); VKCHECK(error);

						submitInfo.pCommandBuffers=&commandBuffers[2];
						error=vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE); VKCHECK(error);
						error=vkQueueWaitIdle(queue); VKCHECK(error);

						if(correctness)
						{
							MatrixDesc const
								*mat_a=matrices+MAT_A,
								*mat_b=matrices+MAT_B,
								*mat_c=matrices+MAT_C,
								*mat_d=matrices+MAT_D;
							int pass=1;
							if(isFloatType(mat_a))
							{
								for (int i=0;i<(int)testCase.M;++i)
								{
									for (int j=0;j<(int)testCase.N;++j)
									{
										float ref=0;
										for (int k=0;k<(int)testCase.K;++k)
											ref+=getDataFloat_c(mat_a, i, k, 0)*getDataFloat_c(mat_b, k, j, testCase.BColMajor);

										ref=alpha*ref+beta*getDataFloat_c(mat_c, i, j, 0);

										float Dij=getDataFloat_c(mat_d, i, j, 0);
										if(ref!=Dij)
										{
											pass=0;
											printf("error %d %d %f != %f\n", i, j, ref, Dij);
										}
									}
								}
							}
							else
							{
								for(uint32_t i=0;i<testCase.M;++i)
								{
									for(uint32_t j=0;j<testCase.N;++j)
									{
										uint32_t ref=0;
										for(uint32_t k=0;k<testCase.K;++k)
											ref+=getDataInt_c(mat_a, i, k, 0)*getDataInt_c(mat_b, k, j, testCase.BColMajor);

										ref=((int)alpha)*ref + ((int)beta)*getDataInt_c(mat_c, i, j, 0);

										uint32_t Dij=getDataInt_c(mat_d, i, j, 0);
										if(ref!=Dij)
										{
											pass=0;
											printf("error %d %d %d != %d\n", i, j, ref, Dij);
										}
									}
								}
							}
							printf("%s\n", pass ? "pass" : "fail");
						}

						// Free the memory/buffers/pipeline for this iteration.
						for (int i = 0; i < NUM_MATS; ++i)
							destroyMatrixDesc(device, matrices+i);
						vkDestroyPipeline(device, pipeline, 0);

						if (maxPerfThisIter < tflops)
							maxPerfThisIter = tflops;

						//Stop this iteration (increasing tile size) if we've gotten to the point where performance is decreasing.
						//This usually means the tile no longer fits in register file.
						if (!correctness && tflops < maxPerfThisIter / 2 && tt == TT_TILED)
							break;
					} // bcolmajor
				} // TILE_N_size
			} // TILE_M_size

			vkDestroyShaderModule(device, shaderModule, 0);
		} // numCooperativeMatrixProperties
	} // TT_COUNT
#endif


#if 0
	VkPhysicalDeviceMemoryProperties memprop={0};
	vkGetPhysicalDeviceMemoryProperties(pdevice, &memprop);
	VkMemoryPropertyFlags reqmemprop;
	int memtypeidx=-1;
	for(int k=0;k<memprop.memoryTypes;++k)
	{
		if(memprop.memoryTypes[k].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
		}
	}
	unsigned nqfamilies=0;
	VkQueueFamilyProperties *qfamprops=0;
	vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &nqfamilies, 0);
	qfamprops=(VkQueueFamilyProperties*)malloc(nqfamilies*sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(pdevice, &nqfamilies, qfamprops);
	int qfamidx=-1;
	for(int k=0;k<nqfamilies;++k)
	{
		if(qfamprops[k].queueFlags&VK_QUEUE_COMPUTE_BIT)
		{
			qfamidx=k;
			break;
		}
	}
	ASSERT_MSG(qfamidx!=-1, "No compute queues on this device");
	printf("Queue Family #%d (of %d) has compute capability\n", qfamidx, nqfamilies);

	float priority[]={0.5f};
	VkDeviceQueueCreateInfo qcreateinfo=
	{
		VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0,
		qfamidx,	//queue family index
		1,			//count of created queues
		priority,	//priorities of created queues
	};
	VkDeviceCreateInfo devcreateinfo=
	{
		VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0,
		1, &qcreateinfo,
		COUNTOF(layernames), layernames,//layers
		0, 0,//extensions
		0,//VkPhysicalDeviceFeatures
	};
	error=vkCreateDevice(pdevice, &devcreateinfo, 0, &device);	VKCHECK(error);


	//create buffers
	VkBufferCreateInfo bufcreateinfo=
	{
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0,
		10*sizeof(float),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	};
#endif

	finish_vk();//
	printf("Done.\n");
	pause();
	exit(0);
}