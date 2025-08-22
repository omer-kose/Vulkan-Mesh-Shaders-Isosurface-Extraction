#include "MarchingCubesIndirectPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline MarchingCubesIndirectPass::GraphicsPipeline = VK_NULL_HANDLE;
VkPipelineLayout MarchingCubesIndirectPass::GraphicsPipelineLayout = VK_NULL_HANDLE;
VkPipeline MarchingCubesIndirectPass::ComputePipeline = VK_NULL_HANDLE;
VkPipelineLayout MarchingCubesIndirectPass::ComputePipelineLayout = VK_NULL_HANDLE;
VkDescriptorSet MarchingCubesIndirectPass::MCDescriptorSet = VK_NULL_HANDLE;
VkDescriptorSetLayout MarchingCubesIndirectPass::MCDescriptorSetLayout= VK_NULL_HANDLE;
VkDescriptorSet MarchingCubesIndirectPass::ComputeDescriptorSet = VK_NULL_HANDLE;
VkDescriptorSetLayout MarchingCubesIndirectPass::ComputeDescriptorSetLayout = VK_NULL_HANDLE;
AllocatedBuffer MarchingCubesIndirectPass::MCLookupTableBuffer = {};
MarchingCubesIndirectPass::MCPushConstants MarchingCubesIndirectPass::PushConstants = {};

template<typename T>
T ceilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

void MarchingCubesIndirectPass::Init(VulkanEngine* engine)
{
    // Init the resources
    size_t lookupTableSize = sizeof(MarchingCubesLookupTable);
	MCLookupTableBuffer = engine->createAndUploadGPUBuffer(lookupTableSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, (void*)MarchingCubesLookupTable);

    // Init the pipeline
    // Load the shaders
    VkShaderModule computeShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes_indirect/marching_cubes_comp.spv", &computeShader))
    {
        fmt::println("Error when building marching cubes indirect compute shader");
    }

    VkShaderModule taskShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes_indirect/marching_cubes_task.spv", &taskShader))
    {
        fmt::println("Error when building marching cubes indirect task shader");
    }

    VkShaderModule meshShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes_indirect/marching_cubes_mesh.spv", &meshShader))
    {
        fmt::println("Error when building marching cubes indirect mesh shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/marching_cubes_indirect/marching_cubes_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building marching cubes indirect fragment shader");
    }

    // Push Constant
    VkPushConstantRange pcRange{ .stageFlags = 0, .offset = 0, .size = sizeof(MCPushConstants)};
    // Descriptors
    // Create Graphics Descriptor Sets
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // MC Table (only used in the Mesh Shader)
    MCDescriptorSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_MESH_BIT_EXT);
    MCDescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, MCDescriptorSetLayout);
    DescriptorWriter writer;
    writer.writeBuffer(0, MCLookupTableBuffer.buffer, lookupTableSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.updateSet(engine->device, MCDescriptorSet);
    // Create Compute Descriptor Sets
    layoutBuilder.clear();
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Depth Pyramid
    ComputeDescriptorSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_COMPUTE_BIT);
    ComputeDescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, ComputeDescriptorSetLayout);
    std::vector<VkDescriptorSetLayout> layouts = { engine->getSceneDescriptorLayout(), VK_NULL_HANDLE };

    // Create The Compute Pipeline
    ComputePipelineBuilder computePipelineBuilder;
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layouts[1] = ComputeDescriptorSetLayout;
    std::tie(ComputePipelineLayout, ComputePipeline) = computePipelineBuilder.buildPipeline(engine->device, computeShader, {pcRange}, layouts);

    // Create The Graphics Pipeline
    pcRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    layouts[1] = MCDescriptorSetLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    pipelineLayoutInfo.setLayoutCount = layouts.size();
    pipelineLayoutInfo.pSetLayouts = layouts.data();

    VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &GraphicsPipelineLayout));

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

    pipelineBuilder.pipelineLayout = GraphicsPipelineLayout;
    GraphicsPipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, computeShader, nullptr);
    vkDestroyShaderModule(engine->device, taskShader, nullptr);
    vkDestroyShaderModule(engine->device, meshShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
}

void MarchingCubesIndirectPass::ExecuteComputePass(VulkanEngine* engine, VkCommandBuffer cmd, uint32_t numActiveChunks)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipeline);
    vkCmdPushConstants(cmd, ComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MCPushConstants), &PushConstants);
    // bind descriptors
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelineLayout, 1, 1, &ComputeDescriptorSet, 0, nullptr);

    vkCmdDispatch(cmd, ceilDiv(numActiveChunks, 128u), 1, 1);
}

void MarchingCubesIndirectPass::ExecuteGraphicsPass(VulkanEngine* engine, VkCommandBuffer cmd, VkBuffer indirectCommandBuffer)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline);
    // set dynamic state
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    // push constants
    vkCmdPushConstants(cmd, GraphicsPipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(MCPushConstants), &PushConstants);
    // bind descriptors
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 1, 1, &MCDescriptorSet, 0, nullptr);

    vkCmdDrawMeshTasksIndirectEXT(cmd, indirectCommandBuffer, 0, 1, 0);
}

void MarchingCubesIndirectPass::SetGridShellSizes(glm::uvec3& gridSize, glm::uvec3& shellSize)
{
    PushConstants.mcSettings.gridSize = gridSize;
    PushConstants.mcSettings.shellSize = shellSize;
}

void MarchingCubesIndirectPass::SetInputIsovalue(float isovalue)
{
    PushConstants.mcSettings.isovalue = isovalue;
}

void MarchingCubesIndirectPass::SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler)
{
    DescriptorWriter writer;
    writer.writeImage(0, depthPyramidView, depthPyramidSampler, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.updateSet(engine->device, ComputeDescriptorSet);
}

void MarchingCubesIndirectPass::SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& chunkDrawDataBufferAddress, const VkDeviceAddress& activeChunkIndicesBufferAddress, const VkDeviceAddress& drawChunkCountBufferAddress)
{
    PushConstants.chunkMetadataBufferAddress = chunkMetadataBufferAddress;
    PushConstants.chunkDrawDataBufferAddress = chunkDrawDataBufferAddress;
    PushConstants.activeChunkIndicesBufferAddress = activeChunkIndicesBufferAddress;
    PushConstants.drawChunkCountBufferAddress = drawChunkCountBufferAddress;
}

void MarchingCubesIndirectPass::SetNumActiveChunks(uint32_t numActiveChunks)
{
    PushConstants.numActiveChunks = numActiveChunks;
}

void MarchingCubesIndirectPass::SetCameraZNear(float zNear)
{
    PushConstants.zNear = zNear;
}

void MarchingCubesIndirectPass::SetDepthPyramidSizes(uint32_t depthPyramidWidth, uint32_t depthPyramidHeight)
{
    PushConstants.depthPyramidWidth = depthPyramidWidth;
    PushConstants.depthPyramidHeight = depthPyramidHeight;
}

void MarchingCubesIndirectPass::ClearResources(VulkanEngine* engine)
{
    vkDestroyDescriptorSetLayout(engine->device, MCDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->device, ComputeDescriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(engine->device, GraphicsPipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, GraphicsPipeline, nullptr);\
    vkDestroyPipelineLayout(engine->device, ComputePipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, ComputePipeline, nullptr);
    engine->destroyBuffer(MCLookupTableBuffer);
}
