#pragma once
#include <Core/vk_types.h>

#include "MarchingCubesLookup.h"

#include "glm/glm.hpp"

class VulkanEngine;

class MarchingCubesPass
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
		MCSettings mcSettings; // This is directly controlled by user. Pass only uses the already written values does not writes onto it. Updating the settings is done via UpdateMCSettings function
		VkDeviceAddress voxelBufferDeviceAddress;
		// Positional Limits of the Grid
		glm::vec3 lowerCornerPos;
		glm::vec3 upperCornerPos;
	};

public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd);
	static void UpdateMCSettings(const MCSettings& mcSettings); // Updates the mcSettings in the Push Constants
	static void SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler);
	static void SetVoxelBufferDeviceAddress(const VkDeviceAddress& voxelBufferDeviceAddress);
	static void SetGridCornerPositions(const glm::vec3& lowerCornerPos, const glm::vec3& upperCornerPos);
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