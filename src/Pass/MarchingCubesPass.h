#pragma once
#include <Core/vk_types.h>

#include "MarchingCubesLookup.h"

class VulkanEngine;
struct RenderObject;

class MarchingCubesPass
{
public:
	static void Init(VulkanEngine* engine);
	static void Execute(VulkanEngine* engine, VkCommandBuffer& cmd);
	static void Update();
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	static VkDescriptorSet MCDescriptorSet; // set=1
	// Resources
	static AllocatedBuffer MCLookupTableBuffer; // set=1 binding=0 uniform buffer
};