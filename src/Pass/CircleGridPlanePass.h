#pragma once
#include <Core/vk_types.h>

class VulkanEngine;
struct RenderObject;

class CircleGridPlanePass
{
public:
	struct GridPlanePushConstants
	{
		float planeHeight;
	};
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer cmd);
	static void Update();
	static void ClearResources(VulkanEngine* engine);
	static void SetPlaneHeight(float height);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	static GridPlanePushConstants PushConstants;
};