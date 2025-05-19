#pragma once
#include <Core/vk_types.h>

class VulkanEngine;
struct RenderObject;

class GLTFMetallicPass
{
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer& cmd);
	static void Update();
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline OpaquePipeline;
	static VkPipeline TransparentPipeline;
	static VkPipelineLayout PipelineLayout; // both transparent and opaque objects use the same pipeline layout
};