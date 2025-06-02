#pragma once
#include <Core/vk_types.h>

#include "MarchingCubesLookup.h"

#include "glm/glm.hpp"

class VulkanEngine;
struct RenderObject;

class MarchingCubesPassSDF
{
public:
	struct MCSettings
	{
		glm::uvec3 gridSize; // Either determined by the input data or the user if a custom SDF is used (such as a noise function)
	};
public:
	static void Init(VulkanEngine* engine, const MCSettings& mcSettings_in);
	static void Execute(VulkanEngine* engine, VkCommandBuffer& cmd);
	static void Update();
	static void ClearResources(VulkanEngine* engine);
private:
	static VkPipeline Pipeline;
	static VkPipelineLayout PipelineLayout;
	static VkDescriptorSet MCDescriptorSet; // set=1
	// Resources
	static AllocatedBuffer MCLookupTableBuffer; // set=1 binding=0 uniform buffer
	static AllocatedBuffer MCSettingsBuffer; // set=1 binding=1 uniform buffer 
	// Misc.
	static MCSettings Settings; // to keep track of settings + utility (like gridSize is used while dispatching)
};