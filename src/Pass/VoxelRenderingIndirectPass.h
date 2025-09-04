#pragma once
#include <Core/vk_types.h>

#include "glm/glm.hpp"

class VulkanEngine;

class VoxelRenderingIndirectPass
{
public:
	/*
	Chunk metadata unique to that chunk. Common values are stored in MCSettings above
	*/
	struct ChunkMetadata
	{
		// Positional Limits of the Grid
		glm::vec3 lowerCornerPos;
		glm::vec3 upperCornerPos;
		VkDeviceAddress voxelBufferDeviceAddress; // base address of the voxels of the chunk in the Voxel Buffer
	};
	/*
		Necessary information for the task and mesh shaders to be able to fetch required chunk data for dispatch. This will be filled by the compute shader for each task shader dispatch.

		Each task shader invocation has a unique ChunkDrawData entry.
	*/
	struct ChunkDrawData
	{
		uint32_t chunkID; // ID of the chunk in the ChunkMetadata array
		// Note: This actually could be computed on the fly: workGroupID % (numGroupsPerChunk). But for ease in debugging, I will keep this explicit.
		uint32_t localWorkgroupID; // Explicitly assign a local work group ID working on that chunk. In other words, this is the id of the block that task shader will work on in the chunk. Range: [0, numGroupsPerChunk - 1]
	};
	struct VoxelPushConstants
	{
		glm::uvec3 chunkSize;
		glm::uvec3 shellSize; // For chunks a shell with +2 on right-bottom-front boundaries for correct computation. For voxel rendering, only +1 is enough to check neighbor occupation but I use the same chuking strategy for both MC and Voxel rendering
		glm::vec3 voxelSize; // Size of a singular voxel. All the voxels are uniformly shaped
		uint32_t numChunks;
		float zNear;
		uint32_t depthPyramidWidth;
		uint32_t depthPyramidHeight;
		VkDeviceAddress chunkMetadataBufferAddress;
		VkDeviceAddress chunkDrawDataBufferAddress;
		VkDeviceAddress drawChunkCountBufferAddress;
	};

public:
	static void Init(VulkanEngine* engine);
	static void ExecuteComputePass(VulkanEngine* engine, VkCommandBuffer cmd, uint32_t numBlocks);
	static void ExecuteGraphicsPass(VulkanEngine* engine, VkCommandBuffer cmd, VkBuffer indirectCommandBuffer);
	// Push Constant Set Functions
	static void SetGridShellSizes(const glm::uvec3& chunkSize, const glm::uvec3& shellSize);
	static void SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler);
	static void SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& chunkDrawDataBufferAddress, const VkDeviceAddress& drawChunkCountBufferAddress);
	static void SetVoxelSize(const glm::vec3& voxelSize);
	static void SetNumChunks(uint32_t numChunks);
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