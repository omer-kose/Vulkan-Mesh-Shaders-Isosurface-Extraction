#pragma once
#include <Core/vk_types.h>

#include "glm/glm.hpp"

class VulkanEngine;

class VoxelRenderingIndirectSVOPass
{
public:
	/*
		Necessary information for the task and mesh shaders to be able to fetch required chunk data for dispatch. This will be filled by the compute shader for each task shader dispatch.

		Each task shader invocation has a unique NodeDrawData entry
	*/
	struct NodeDrawData
	{
		uint32_t nodeID; // ID of the chunk in the ChunkMetadata array
	};
	struct VoxelPushConstants
	{
		uint32_t numActiveNodes; // Number of nodes to process this frame. Not the total number of nodes. Practically active size of the nodeIndices bufer
		uint32_t leafLevel; // TODO: Try making this uint8_t. Should work I believe
		float zNear;
		uint32_t depthPyramidWidth;
		uint32_t depthPyramidHeight;
		VkDeviceAddress svoNodeGPUBufferAddress;
		VkDeviceAddress brickBufferAddress;
		VkDeviceAddress nodeDrawDataBufferAddress;
		VkDeviceAddress drawNodeCountBufferAddress;
		VkDeviceAddress activeNodeIndicesBufferAddress;
	};

public:
	static void Init(VulkanEngine* engine);
	static void ExecuteComputePass(VulkanEngine* engine, VkCommandBuffer cmd);
	static void ExecuteGraphicsPass(VulkanEngine* engine, VkCommandBuffer cmd, VkBuffer indirectCommandBuffer);
	// Push Constant Set Functions
	static void SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler);
	static void SetBufferAddresses(const VkDeviceAddress& svoNodeGPUBufferAddress, const VkDeviceAddress& brickBufferAddress, const VkDeviceAddress& nodeDrawDataBufferAddress, const VkDeviceAddress& drawNodeCountBufferAddress, const VkDeviceAddress& activeNodeIndicesBufferAddress);
	static void SetNumActiveNodes(uint32_t numNodes);
	static void SetLeafLevel(uint32_t leafLevel);
	static void SetCameraZNear(float zNear);
	static void SetDepthPyramidSizes(uint32_t depthPyramidWidth, uint32_t depthPyramidHeight);
	static void SetColorPaletteBinding(VulkanEngine* engine, VkBuffer colorPaletteBuffer, size_t bufferSize);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline GraphicsPipeline;
	static VkPipelineLayout GraphicsPipelineLayout;
	static VkPipeline ComputePipeline;
	static VkPipelineLayout ComputePipelineLayout;
	static VkDescriptorSet GraphicsDescriptorSet;  // set=1 containing color palette
	static VkDescriptorSetLayout GraphicsDescriptorSetLayout; // Vulkan complains when layout is deleted. It is a false positive but anyways. (it complains in only some passes I have no idea why yet)
	static VkDescriptorSet ComputeDescriptorSet; // set=1 containing the depth image for the compute pass
	static VkDescriptorSetLayout ComputeDescriptorSetLayout; // Vulkan complains when layout is deleted. It is a false positive but anyways. (it complains in only some passes I have no idea why yet)
	// Resources
	static VoxelPushConstants PushConstants; // Engine can modify this via Update functions
};