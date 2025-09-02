#pragma once

#include <Core/vk_types.h>

class VulkanEngine;

class OccluderPrePass
{
public:
	struct OccluderPushConstants
	{
		VkDeviceAddress chunkMetadataBufferAddress;
		VkDeviceAddress chunkDrawDataBufferAddress;
		glm::uvec3 chunkSize;
	};
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd, VkBuffer indirectCommandBuffer);
	static void SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& chunkDrawDataBufferAddress);
	static void SetChunkSize(const glm::uvec3& chunkSize);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	// Resources
	static OccluderPushConstants PushConstants; // MC Settings are kept track of via PushConstants. Engine can modify this via Update functions
};