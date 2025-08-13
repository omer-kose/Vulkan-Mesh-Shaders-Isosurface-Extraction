#include "HZBDownSamplePass.h"

#include <Core/vk_engine.h>
#include <Core/vk_images.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_barriers.h>

VkPipeline HZBDownSamplePass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout HZBDownSamplePass::PipelineLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout HZBDownSamplePass::DescriptorSetLayout = VK_NULL_HANDLE;
DescriptorWriter HZBDownSamplePass::Writer = {};
HZBDownSamplePass::HZBDownSamplePushConstants HZBDownSamplePass::PushConstants = {};
// Resources
AllocatedImage HZBDownSamplePass::DepthPyramid = {}; // Previous frame's depth image with mips being manually downsampled versions with a 2x2 min kernel.
std::vector<VkImageView> HZBDownSamplePass::DepthPyramidMips; // The image contains all the mips but views must be taken manually to be able to bind them with descriptors while processing them
VkSampler HZBDownSamplePass::DepthPyramidImageSampler = VK_NULL_HANDLE;
uint32_t HZBDownSamplePass::DepthPyramidWidth;
uint32_t HZBDownSamplePass::DepthPyramidHeight;
uint32_t HZBDownSamplePass::DepthPyramidLevels;

uint32_t previousPow2(uint32_t v)
{
	uint32_t r = 1;

	while(r * 2 < v)
		r *= 2;

	return r;
}

uint32_t getImageMipLevels(uint32_t width, uint32_t height)
{
	return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

template<typename T>
T ceilDiv(T x, T y)
{
	return (x + y - 1) / y;
}

void depthTransition(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout, VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT)
{
	VkImageMemoryBarrier2 imageBarrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .pNext = nullptr };

	imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
	imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

	imageBarrier.oldLayout = currentLayout;
	imageBarrier.newLayout = newLayout;

	imageBarrier.subresourceRange = vkinit::image_subresource_range(aspectMask);
	imageBarrier.image = image;

	VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .pNext = nullptr };
	depInfo.imageMemoryBarrierCount = 1;
	depInfo.pImageMemoryBarriers = &imageBarrier;

	vkCmdPipelineBarrier2(cmd, &depInfo);
}

void HZBDownSamplePass::Init(VulkanEngine* engine)
{
	// Create the Depth Pyramid image and the views to the mips
	DepthPyramidWidth = previousPow2(engine->depthImage.imageExtent.width);
	DepthPyramidHeight = previousPow2(engine->depthImage.imageExtent.height);
	DepthPyramidLevels = getImageMipLevels(DepthPyramidWidth, DepthPyramidHeight);

	// createImage automatically creates correct amount of mipLevels (uses the same formula as above)
	DepthPyramid = engine->createImage(VkExtent3D{DepthPyramidWidth, DepthPyramidHeight, 1}, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true);
	DepthPyramidMips.resize(DepthPyramidLevels);

	for(size_t i = 0; i < DepthPyramidLevels; ++i)
	{
		DepthPyramidMips[i] = engine->createImageView(DepthPyramid.image, VK_FORMAT_R32_SFLOAT, i, 1);
	}

	DepthPyramidImageSampler = engine->createImageSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_REDUCTION_MODE_MIN);

	// Init the pipeline
	// Load the shaders
	VkShaderModule computeShader;
	if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/hzb_downsample/hzb_downsample_comp.spv", &computeShader))
	{
		fmt::println("Error when building HZB Downsample compute shader");
	}

	// Push Constant (only contains the size of the mip level that is going to be written on)
	VkPushConstantRange pcRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT , .offset = 0, .size = sizeof(PushConstants) };

	// Set descriptor sets
	DescriptorLayoutBuilder layoutBuilder;
	layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // upper mip level to be sampled from
	layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // Output mip level to be written on
	DescriptorSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
	// Allocate the descriptor set and update
	//DescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, DescriptorSetLayout);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pcRange;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &DescriptorSetLayout;

	VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

	VkPipelineShaderStageCreateInfo shaderStageInfo = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShader);
	VkComputePipelineCreateInfo pipelineInfo = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr };
	pipelineInfo.layout = PipelineLayout;
	pipelineInfo.stage = shaderStageInfo;

	VK_CHECK(vkCreateComputePipelines(engine->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Pipeline));

	vkDestroyShaderModule(engine->device, computeShader, nullptr);

	// TODO: Could be created in a specific layout I believe
	engine->immediateSubmit([&](VkCommandBuffer cmd) {
		// Transition DepthPyramid image to correct format
		vkutil::transitionImage(cmd, DepthPyramid.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
	});
}

void HZBDownSamplePass::Execute(VulkanEngine* engine, VkCommandBuffer cmd)
{
	// Transition the images to proper layouts
	VkImageMemoryBarrier2 depthBarriers[] = {
		vkutil::imageBarrier(engine->depthImage.image,
			VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_IMAGE_ASPECT_DEPTH_BIT),
		vkutil::imageBarrier(DepthPyramid.image,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL)
	};

	vkutil::pipelineBarrier(cmd, 0, 0, nullptr, 2, depthBarriers);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Pipeline);
	
	// Hierarchically downsample from base image to all the mips
	for(uint32_t i = 0; i < DepthPyramidLevels; ++i)
	{
		VkImageView inImage = i != 0 ? DepthPyramidMips[i - 1] : engine->depthImage.imageView;
		VkImageLayout inLayout = i != 0 ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		// Update descriptor sets
		Writer.clear();
		// Source level
		Writer.writeImage(0, inImage, DepthPyramidImageSampler, inLayout, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
		// Destination Level
		Writer.writeImage(1, DepthPyramidMips[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
		Writer.pushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, PipelineLayout, 0);

		// Update and push constants
		uint32_t levelWidth = std::max(1u, DepthPyramidWidth >> i);
		uint32_t levelHeight = std::max(1u, DepthPyramidHeight >> i);

		PushConstants.outImageSize = glm::vec2(levelWidth, levelHeight);
		vkCmdPushConstants(cmd, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HZBDownSamplePushConstants), &PushConstants);

		vkCmdDispatch(cmd, ceilDiv(levelWidth, 32u), ceilDiv(levelHeight, 32u), 1u);

		VkImageMemoryBarrier2 reduceBarrier = vkutil::imageBarrier(DepthPyramid.image,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
			VK_IMAGE_ASPECT_COLOR_BIT, i, 1);

		vkutil::pipelineBarrier(cmd, 0, 0, nullptr, 1, &reduceBarrier);
	}

	// Transition back to original layouts;
	VkImageMemoryBarrier2 depthWriteBarrier = vkutil::imageBarrier(engine->depthImage.image,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		VK_IMAGE_ASPECT_DEPTH_BIT);

	vkutil::pipelineBarrier(cmd, 0, 0, nullptr, 1, &depthWriteBarrier);

}

void HZBDownSamplePass::Update()
{
}

VkImageView HZBDownSamplePass::GetDepthPyramidImageView()
{
	return DepthPyramid.imageView;
}

VkSampler HZBDownSamplePass::GetDepthPyramidSampler()
{
	return DepthPyramidImageSampler;
}

uint32_t HZBDownSamplePass::GetDepthPyramidWidth()
{
	return DepthPyramidWidth;
}

uint32_t HZBDownSamplePass::GetDepthPyramidHeight()
{
	return DepthPyramidHeight;
}

void HZBDownSamplePass::ClearResources(VulkanEngine* engine)
{
	vkDestroyDescriptorSetLayout(engine->device, DescriptorSetLayout, nullptr);
	vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
	vkDestroyPipeline(engine->device, Pipeline, nullptr);
	engine->destroyImage(DepthPyramid);
	for(uint32_t i = 0; i < DepthPyramidLevels; ++i)
	{
		vkDestroyImageView(engine->device, DepthPyramidMips[i], nullptr);
	}
	vkDestroySampler(engine->device, DepthPyramidImageSampler, nullptr);
}
