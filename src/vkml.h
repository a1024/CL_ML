#pragma once
#ifndef INC_VKML_H
#define INC_VKML_H
#define VK_NO_PROTOTYPES
#include<vulkan/vulkan.h>
#ifdef __cplusplus
extern "C"
{
#endif
	
#define VKFUNCLIST1\
	VKFUNC(vkGetInstanceProcAddr)

#define VKFUNCLIST2\
	VKFUNC(vkCreateInstance)\
	VKFUNC(vkEnumerateInstanceExtensionProperties)\
	VKFUNC(vkEnumerateInstanceLayerProperties)\
	VKFUNC(vkEnumerateInstanceVersion)

#define VKFUNCLIST3\
	VKFUNC(vkAllocateCommandBuffers)/*Vulkan 1.0*/\
	VKFUNC(vkAllocateDescriptorSets)\
	VKFUNC(vkAllocateMemory)\
	VKFUNC(vkBeginCommandBuffer)\
	VKFUNC(vkBindBufferMemory)\
	VKFUNC(vkBindImageMemory)/**/\
	VKFUNC(vkCmdBeginQuery)\
	VKFUNC(vkCmdBeginRenderPass)\
	VKFUNC(vkCmdBindDescriptorSets)\
	VKFUNC(vkCmdBindIndexBuffer)\
	VKFUNC(vkCmdBindPipeline)\
	VKFUNC(vkCmdBindVertexBuffers)\
	VKFUNC(vkCmdBlitImage)\
	VKFUNC(vkCmdClearAttachments)\
	VKFUNC(vkCmdClearColorImage)\
	VKFUNC(vkCmdClearDepthStencilImage)\
	VKFUNC(vkCmdCopyBuffer)\
	VKFUNC(vkCmdCopyBufferToImage)\
	VKFUNC(vkCmdCopyImage)\
	VKFUNC(vkCmdCopyImageToBuffer)\
	VKFUNC(vkCmdCopyQueryPoolResults)\
	VKFUNC(vkCmdDispatch)\
	VKFUNC(vkCmdDispatchIndirect)\
	VKFUNC(vkCmdDraw)\
	VKFUNC(vkCmdDrawIndexed)\
	VKFUNC(vkCmdDrawIndexedIndirect)\
	VKFUNC(vkCmdDrawIndirect)\
	VKFUNC(vkCmdEndQuery)\
	VKFUNC(vkCmdEndRenderPass)\
	VKFUNC(vkCmdExecuteCommands)\
	VKFUNC(vkCmdFillBuffer)\
	VKFUNC(vkCmdNextSubpass)\
	VKFUNC(vkCmdPipelineBarrier)\
	VKFUNC(vkCmdPushConstants)\
	VKFUNC(vkCmdResetEvent)\
	VKFUNC(vkCmdResetQueryPool)\
	VKFUNC(vkCmdResolveImage)\
	VKFUNC(vkCmdSetBlendConstants)\
	VKFUNC(vkCmdSetDepthBias)\
	VKFUNC(vkCmdSetDepthBounds)\
	VKFUNC(vkCmdSetEvent)\
	VKFUNC(vkCmdSetLineWidth)\
	VKFUNC(vkCmdSetScissor)\
	VKFUNC(vkCmdSetStencilCompareMask)\
	VKFUNC(vkCmdSetStencilReference)\
	VKFUNC(vkCmdSetStencilWriteMask)\
	VKFUNC(vkCmdSetViewport)\
	VKFUNC(vkCmdUpdateBuffer)\
	VKFUNC(vkCmdWaitEvents)\
	VKFUNC(vkCmdWriteTimestamp)\
	VKFUNC(vkCreateBuffer)\
	VKFUNC(vkCreateBufferView)\
	VKFUNC(vkCreateCommandPool)\
	VKFUNC(vkCreateComputePipelines)\
	VKFUNC(vkCreateDescriptorPool)\
	VKFUNC(vkCreateDescriptorSetLayout)\
	VKFUNC(vkCreateDevice)\
	VKFUNC(vkCreateEvent)\
	VKFUNC(vkCreateFence)\
	VKFUNC(vkCreateFramebuffer)\
	VKFUNC(vkCreateGraphicsPipelines)\
	VKFUNC(vkCreateImage)\
	VKFUNC(vkCreateImageView)\
	VKFUNC(vkCreatePipelineCache)\
	VKFUNC(vkCreatePipelineLayout)\
	VKFUNC(vkCreateQueryPool)\
	VKFUNC(vkCreateRenderPass)\
	VKFUNC(vkCreateSampler)\
	VKFUNC(vkCreateSemaphore)\
	VKFUNC(vkCreateShaderModule)\
	VKFUNC(vkDestroyBuffer)\
	VKFUNC(vkDestroyBufferView)\
	VKFUNC(vkDestroyCommandPool)\
	VKFUNC(vkDestroyDescriptorPool)\
	VKFUNC(vkDestroyDescriptorSetLayout)\
	VKFUNC(vkDestroyDevice)\
	VKFUNC(vkDestroyEvent)\
	VKFUNC(vkDestroyFence)\
	VKFUNC(vkDestroyFramebuffer)\
	VKFUNC(vkDestroyImage)\
	VKFUNC(vkDestroyImageView)\
	VKFUNC(vkDestroyInstance)\
	VKFUNC(vkDestroyPipeline)\
	VKFUNC(vkDestroyPipelineCache)\
	VKFUNC(vkDestroyPipelineLayout)\
	VKFUNC(vkDestroyQueryPool)\
	VKFUNC(vkDestroyRenderPass)\
	VKFUNC(vkDestroySampler)\
	VKFUNC(vkDestroySemaphore)\
	VKFUNC(vkDestroyShaderModule)\
	VKFUNC(vkDeviceWaitIdle)\
	VKFUNC(vkEndCommandBuffer)\
	VKFUNC(vkEnumerateDeviceExtensionProperties)\
	VKFUNC(vkEnumerateDeviceLayerProperties)\
	VKFUNC(vkEnumeratePhysicalDevices)\
	VKFUNC(vkFlushMappedMemoryRanges)\
	VKFUNC(vkFreeCommandBuffers)\
	VKFUNC(vkFreeDescriptorSets)\
	VKFUNC(vkFreeMemory)\
	VKFUNC(vkGetBufferMemoryRequirements)\
	VKFUNC(vkGetDeviceMemoryCommitment)/**/\
	VKFUNC(vkGetDeviceProcAddr)\
	VKFUNC(vkGetDeviceQueue)\
	VKFUNC(vkGetEventStatus)\
	VKFUNC(vkGetFenceStatus)\
	VKFUNC(vkGetImageMemoryRequirements)\
	VKFUNC(vkGetImageSparseMemoryRequirements)/**/\
	VKFUNC(vkGetImageSubresourceLayout)\
	VKFUNC(vkGetPhysicalDeviceFeatures)\
	VKFUNC(vkGetPhysicalDeviceFormatProperties)\
	VKFUNC(vkGetPhysicalDeviceImageFormatProperties)\
	VKFUNC(vkGetPhysicalDeviceMemoryProperties)\
	VKFUNC(vkGetPhysicalDeviceProperties)\
	VKFUNC(vkGetPhysicalDeviceQueueFamilyProperties)\
	VKFUNC(vkGetPhysicalDeviceSparseImageFormatProperties)\
	VKFUNC(vkGetPipelineCacheData)\
	VKFUNC(vkGetQueryPoolResults)\
	VKFUNC(vkGetRenderAreaGranularity)\
	VKFUNC(vkInvalidateMappedMemoryRanges)/**/\
	VKFUNC(vkMapMemory)\
	VKFUNC(vkMergePipelineCaches)\
	VKFUNC(vkQueueBindSparse)\
	VKFUNC(vkQueueSubmit)\
	VKFUNC(vkQueueWaitIdle)\
	VKFUNC(vkResetCommandBuffer)\
	VKFUNC(vkResetCommandPool)\
	VKFUNC(vkResetDescriptorPool)\
	VKFUNC(vkResetEvent)\
	VKFUNC(vkResetFences)\
	VKFUNC(vkSetEvent)\
	VKFUNC(vkUnmapMemory)\
	VKFUNC(vkUpdateDescriptorSets)\
	VKFUNC(vkWaitForFences)\
	VKFUNC(vkBindBufferMemory2)/*Vulkan 1.1*/\
	VKFUNC(vkBindImageMemory2)\
	VKFUNC(vkCmdDispatchBase)\
	VKFUNC(vkCmdSetDeviceMask)\
	VKFUNC(vkCreateDescriptorUpdateTemplate)\
	VKFUNC(vkCreateSamplerYcbcrConversion)\
	VKFUNC(vkDestroyDescriptorUpdateTemplate)\
	VKFUNC(vkDestroySamplerYcbcrConversion)\
	VKFUNC(vkEnumeratePhysicalDeviceGroups)\
	VKFUNC(vkGetBufferMemoryRequirements2)\
	VKFUNC(vkGetDescriptorSetLayoutSupport)\
	VKFUNC(vkGetDeviceGroupPeerMemoryFeatures)\
	VKFUNC(vkGetDeviceQueue2)\
	VKFUNC(vkGetImageMemoryRequirements2)\
	VKFUNC(vkGetImageSparseMemoryRequirements2)\
	VKFUNC(vkGetPhysicalDeviceExternalBufferProperties)\
	VKFUNC(vkGetPhysicalDeviceExternalFenceProperties)\
	VKFUNC(vkGetPhysicalDeviceExternalSemaphoreProperties)\
	VKFUNC(vkGetPhysicalDeviceFeatures2)\
	VKFUNC(vkGetPhysicalDeviceFormatProperties2)\
	VKFUNC(vkGetPhysicalDeviceImageFormatProperties2)\
	VKFUNC(vkGetPhysicalDeviceMemoryProperties2)\
	VKFUNC(vkGetPhysicalDeviceProperties2)\
	VKFUNC(vkGetPhysicalDeviceQueueFamilyProperties2)\
	VKFUNC(vkGetPhysicalDeviceSparseImageFormatProperties2)\
	VKFUNC(vkTrimCommandPool)\
	VKFUNC(vkUpdateDescriptorSetWithTemplate)\
	VKFUNC(vkCmdBeginRenderPass2)/*Vulkan 1.2*/\
	VKFUNC(vkCmdDrawIndexedIndirectCount)\
	VKFUNC(vkCmdDrawIndirectCount)\
	VKFUNC(vkCmdEndRenderPass2)\
	VKFUNC(vkCmdNextSubpass2)\
	VKFUNC(vkCreateRenderPass2)\
	VKFUNC(vkGetBufferDeviceAddress)\
	VKFUNC(vkGetBufferOpaqueCaptureAddress)\
	VKFUNC(vkGetDeviceMemoryOpaqueCaptureAddress)\
	VKFUNC(vkGetSemaphoreCounterValue)\
	VKFUNC(vkResetQueryPool)\
	VKFUNC(vkSignalSemaphore)\
	VKFUNC(vkWaitSemaphores)

//#define VKFUNC(FUNCNAME)	extern PFN_##FUNCNAME p_##FUNCNAME;//I will forget to prepend p_ to every vk function
#define VKFUNC(FUNCNAME)	extern PFN_##FUNCNAME FUNCNAME;
VKFUNCLIST1
VKFUNCLIST2
VKFUNCLIST3
#undef	VKFUNC

void init_vk();

const char* vk_err2str(int err);
#define	VKCHECK(E)	(!(E)||log_error(file, __LINE__, 1, "Vulkan %d: %s", E, vk_err2str(E)))

#ifdef __cplusplus
}
#endif
#endif//INC_VKML_H
