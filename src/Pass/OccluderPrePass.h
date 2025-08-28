#pragma once

#include <Core/vk_types.h>

class VulkanEngine;

class OccluderPrePass
{
public:
	struct OccluderPushConstants
	{
		VkDeviceAddress chunkMetadataBufferAddress;
	};
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd, size_t numChunks);
	static void SetChunkMetadataBufferAddress(const VkDeviceAddress& chunkMetadataBufferAddress);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	// Resources
	static OccluderPushConstants PushConstants; // MC Settings are kept track of via PushConstants. Engine can modify this via Update functions
};