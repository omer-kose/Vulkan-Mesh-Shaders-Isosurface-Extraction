#include "MarchingCubesPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline MarchingCubesPass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout MarchingCubesPass::PipelineLayout = VK_NULL_HANDLE;
VkDescriptorSet MarchingCubesPass::MCDescriptorSet = VK_NULL_HANDLE;
VkDescriptorSetLayout MarchingCubesPass::MCDescriptorSetLayout= VK_NULL_HANDLE;
AllocatedBuffer MarchingCubesPass::MCLookupTableBuffer = {};
MarchingCubesPass::MCPushConstants MarchingCubesPass::PushConstants = {};

template<typename T>
T ceilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

void MarchingCubesPass::Init(VulkanEngine* engine)
{
    // Init the resources
    size_t lookupTableSize = sizeof(MarchingCubesLookupTable);
	MCLookupTableBuffer = engine->createAndUploadGPUBuffer(lookupTableSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, (void*)MarchingCubesLookupTable);

    // Init the pipeline
    // Load the shaders
    VkShaderModule taskShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes/marching_cubes_task.spv", &taskShader))
    {
        fmt::println("Error when building marching cubes task shader");
    }

    VkShaderModule meshShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes/marching_cubes_mesh.spv", &meshShader))
    {
        fmt::println("Error when building marching cubes mesh shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes/marching_cubes_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building marching cubes fragment shader");
    }

    // Push Constant (MC Settings are dynamic and updated via UpdateMCSettings function if needed (needs to be updated at least once of course))
    VkPushConstantRange pcRange{ .stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, .offset = 0, .size = sizeof(MCPushConstants) };

    // Set descriptor sets
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // MC Table (only used in the Mesh Shader)
    layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Depth Pyramid
    MCDescriptorSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT);
    // Allocate the descriptor set and update
    MCDescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, MCDescriptorSetLayout);
    DescriptorWriter writer;
    writer.writeBuffer(0, MCLookupTableBuffer.buffer, lookupTableSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.updateSet(engine->device, MCDescriptorSet);
    // 2 sets: 0 -> Scene Descriptor Set, 1 -> Pass Specific Descriptor Set
    VkDescriptorSetLayout layouts[] = { engine->getSceneDescriptorLayout(), MCDescriptorSetLayout };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;

    VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    // Build the pipeline
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.pushShaderStage(taskShader, VK_SHADER_STAGE_TASK_BIT_EXT);
    pipelineBuilder.pushShaderStage(meshShader, VK_SHADER_STAGE_MESH_BIT_EXT);
    pipelineBuilder.pushShaderStage(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_GREATER);

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);

    pipelineBuilder.pipelineLayout = PipelineLayout;
    Pipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, taskShader, nullptr);
    vkDestroyShaderModule(engine->device, meshShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
}

void MarchingCubesPass::Execute(VulkanEngine* engine, VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    // set dynamic state
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    // push constants
    vkCmdPushConstants(cmd, PipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(MCPushConstants), &PushConstants);
    // bind descriptors
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 1, 1, &MCDescriptorSet, 0, nullptr);

    vkCmdDrawMeshTasksEXT(cmd, ceilDiv(PushConstants.mcSettings.gridSize.x, 4u) * ceilDiv(PushConstants.mcSettings.gridSize.y, 4u) * ceilDiv(PushConstants.mcSettings.gridSize.z, 4u), 1, 1);
}

void MarchingCubesPass::SetGridShellSizes(glm::uvec3& gridSize, glm::uvec3& shellSize)
{
    PushConstants.mcSettings.gridSize = gridSize;
    PushConstants.mcSettings.shellSize = shellSize;
}

void MarchingCubesPass::SetInputIsovalue(float isovalue)
{
    PushConstants.mcSettings.isovalue = isovalue;
}

void MarchingCubesPass::SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler)
{
    DescriptorWriter writer;
    writer.writeImage(1, depthPyramidView, depthPyramidSampler, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.updateSet(engine->device, MCDescriptorSet);
}

void MarchingCubesPass::SetVoxelBufferDeviceAddress(const VkDeviceAddress& voxelBufferDeviceAddress)
{
    PushConstants.voxelBufferDeviceAddress = voxelBufferDeviceAddress; // assigned once as the address does not change
}

void MarchingCubesPass::SetGridCornerPositions(const glm::vec3& lowerCornerPos, const glm::vec3& upperCornerPos)
{
    PushConstants.lowerCornerPos = lowerCornerPos;
    PushConstants.upperCornerPos = upperCornerPos;
}

void MarchingCubesPass::SetCameraZNear(float zNear)
{
    PushConstants.zNear = zNear;
}

void MarchingCubesPass::SetDepthPyramidSizes(uint32_t depthPyramidWidth, uint32_t depthPyramidHeight)
{
    PushConstants.depthPyramidWidth = depthPyramidWidth;
    PushConstants.depthPyramidHeight = depthPyramidHeight;
}

void MarchingCubesPass::ClearResources(VulkanEngine* engine)
{
    vkDestroyDescriptorSetLayout(engine->device, MCDescriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, Pipeline, nullptr);
    engine->destroyBuffer(MCLookupTableBuffer);
}
