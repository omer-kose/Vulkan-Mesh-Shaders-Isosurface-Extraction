#pragma once

#include <Core/vk_types.h>

class VulkanEngine;

class ChunkVisualizationPass
{
public:
	struct ChunkVisPushConstants
	{
		VkDeviceAddress chunkBufferDeviceAddress;
		float inputIsoValue;
	};
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd, size_t numChunks, float lineWidth = 1.0f);
	static void SetChunkBufferDeviceAddress(const VkDeviceAddress& chunkBufferDeviceAddress);
	static void SetInputIsoValue(float isoValue);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	static VkDescriptorSet MCDescriptorSet; // set=1
	// Resources
	static ChunkVisPushConstants PushConstants; // MC Settings are kept track of via PushConstants. Engine can modify this via Update functions
};