#pragma once

#include <Core/vk_types.h>

class VulkanEngine;

class OccluderPrePass
{
public:
	struct OccluderPushConstants
	{
		VkDeviceAddress chunkMetadataBufferAddress;
		VkDeviceAddress activeChunkIndicesBuffer;
		uint32_t numActiveChunks;
	};
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd, size_t numActiveChunks);
	static void SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& activeChunkIndicesBuffer);
	static void SetNumActiveChunks(uint32_t numActiveChunks);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	// Resources
	static OccluderPushConstants PushConstants; // MC Settings are kept track of via PushConstants. Engine can modify this via Update functions
};