#pragma once
#include <Core/vk_types.h>

#include "MarchingCubesLookup.h"

#include "glm/glm.hpp"

class VulkanEngine;

class MarchingCubesIndirectPass
{
public:
	// MC Settings that are sent to the gpu with the Push Constants
	struct MCSettings
	{
		glm::uvec3 gridSize; // Either determined by the input data or the user if a custom SDF is used (such as a noise function)
		glm::uvec3 shellSize; // For chunks a shell with +2 on right-bottom-front boundaries for correct computation. For a non-chunked volume gridSize==shellSize. This is only used for fetching the data correctly with voxelValue()
		float isoValue;
	};
	struct MCPushConstants
	{
		MCSettings mcSettings;
		float zNear;
		uint32_t depthPyramidWidth;
		uint32_t depthPyramidHeight;
		VkDeviceAddress chunkMetadataBufferAddress;
		VkDeviceAddress chunkDrawDataBufferAddress;
		VkDeviceAddress activeChunkIndicesBufferAddress;
		uint32_t numActiveChunks; 
		VkDeviceAddress drawChunkCountBufferAddress;
	};

public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd);
	static void UpdateMCSettings(const MCSettings& mcSettings); // Updates the mcSettings in the Push Constants
	static void SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler);
	static void SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& chunkDrawDataBufferAddress, const VkDeviceAddress& activeChunkIndicesBufferAddress, const VkDeviceAddress& drawChunkCountBufferAddress);
	static void SetCameraZNear(float zNear);
	static void SetDepthPyramidSizes(uint32_t depthPyramidWidth, uint32_t depthPyramidHeight);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	static VkDescriptorSet MCDescriptorSet; // set=1
	static VkDescriptorSetLayout MCDescriptorSetLayout; // Vulkan complains when layout is deleted. It is a false positive but anyways. (it complains in only some passes I have no idea why yet)
	// Resources
	static AllocatedBuffer MCLookupTableBuffer; // set=1 binding=0 uniform buffer
	static MCPushConstants PushConstants; // MC Settings are kept track of via PushConstants. Engine can modify this via Update functions
};