#pragma once
#include <Core/vk_types.h>
#include <Core/vk_descriptors.h>

class VulkanEngine;

class HZBDownSamplePass
{
public:
	struct HZBDownSamplePushConstants
	{
		glm::vec2 outImageSize;
	};
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd);
	static void Update();
	static void CopyDepthImage(VulkanEngine* engine, VkCommandBuffer cmd);
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	static VkDescriptorSet DescriptorSet;
	static VkDescriptorSetLayout DescriptorSetLayout; // I have no idea why but for this sample pass only, vulkan complains when layout is deleted while set is in use.
	static DescriptorWriter Writer;
	static HZBDownSamplePushConstants PushConstants;
	// Resources
	static AllocatedImage DepthPyramid; // Previous frame's depth image with mips being manually downsampled versions with a 2x2 min kernel.
	static std::vector<VkImageView> DepthPyramidMips; // The image contains all the mips but views must be taken manually to be able to bind them with descriptors while processing them
	static VkSampler DepthPyramidImageSampler;
	static uint32_t DepthPyramidWidth;
	static uint32_t DepthPyramidHeight;
	static uint32_t DepthPyramidLevels;
};