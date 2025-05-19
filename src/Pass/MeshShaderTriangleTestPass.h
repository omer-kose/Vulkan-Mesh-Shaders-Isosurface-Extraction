#pragma once
#include <Core/vk_types.h>

class VulkanEngine;
struct RenderObject;

class MeshShaderTriangleTestPass
{
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer& cmd);
	static void Update();
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
};