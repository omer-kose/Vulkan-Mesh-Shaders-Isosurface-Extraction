#include "VoxelRenderingIndirectPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline VoxelRenderingIndirectPass::GraphicsPipeline = VK_NULL_HANDLE;
VkPipelineLayout VoxelRenderingIndirectPass::GraphicsPipelineLayout = VK_NULL_HANDLE;
VkPipeline VoxelRenderingIndirectPass::ComputePipeline = VK_NULL_HANDLE;
VkPipelineLayout VoxelRenderingIndirectPass::ComputePipelineLayout = VK_NULL_HANDLE;
VkDescriptorSet VoxelRenderingIndirectPass::GraphicsDescriptorSet = VK_NULL_HANDLE;
VkDescriptorSetLayout VoxelRenderingIndirectPass::GraphicsDescriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorSet VoxelRenderingIndirectPass::ComputeDescriptorSet = VK_NULL_HANDLE;
VkDescriptorSetLayout VoxelRenderingIndirectPass::ComputeDescriptorSetLayout = VK_NULL_HANDLE;
VoxelRenderingIndirectPass::VoxelPushConstants VoxelRenderingIndirectPass::PushConstants = {};

template<typename T>
T ceilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

void VoxelRenderingIndirectPass::Init(VulkanEngine* engine)
{
    // Init the pipeline
    // Load the shaders
    VkShaderModule computeShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/voxel_rendering/voxel_rendering_comp.spv", &computeShader))
    {
        fmt::println("Error when building voxel rendering indirect compute shader");
    }

    VkShaderModule taskShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/voxel_rendering/voxel_rendering_task.spv", &taskShader))
    {
        fmt::println("Error when building voxel rendering indirect task shader");
    }

    VkShaderModule meshShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/voxel_rendering/voxel_rendering_mesh.spv", &meshShader))
    {
        fmt::println("Error when building voxel rendering indirect mesh shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/voxel_rendering/voxel_rendering_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building voxel rendering indirect fragment shader");
    }

    // Push Constant
    VkPushConstantRange pcRange{ .stageFlags = 0, .offset = 0, .size = sizeof(VoxelPushConstants) };
    // Descriptors
    // Create Graphics Descriptor Sets
    DescriptorLayoutBuilder layoutBuilder;
    VkDescriptorSetLayout sceneDescriptorLayout = engine->getSceneDescriptorLayout();
    
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // Color palette
    GraphicsDescriptorSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_FRAGMENT_BIT);
    GraphicsDescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, GraphicsDescriptorSetLayout);
    // Create Compute Descriptor Sets
    layoutBuilder.clear();
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // Depth Pyramid
    ComputeDescriptorSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_COMPUTE_BIT);
    ComputeDescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, ComputeDescriptorSetLayout);

    // Create The Compute Pipeline
    ComputePipelineBuilder computePipelineBuilder;
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    std::tie(ComputePipelineLayout, ComputePipeline) = computePipelineBuilder.buildPipeline(engine->device, computeShader, { pcRange }, { sceneDescriptorLayout, ComputeDescriptorSetLayout });

    // Create The Graphics Pipeline
    pcRange.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    VkDescriptorSetLayout graphicsSetLayouts[] = { sceneDescriptorLayout, GraphicsDescriptorSetLayout };
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = graphicsSetLayouts;

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

void VoxelRenderingIndirectPass::ExecuteComputePass(VulkanEngine* engine, VkCommandBuffer cmd, uint32_t numBlocks)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipeline);
    vkCmdPushConstants(cmd, ComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VoxelPushConstants), &PushConstants);
    // bind descriptors
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelineLayout, 1, 1, &ComputeDescriptorSet, 0, nullptr);

    vkCmdDispatch(cmd, ceilDiv(numBlocks, 128u), 1, 1);
}

void VoxelRenderingIndirectPass::ExecuteGraphicsPass(VulkanEngine* engine, VkCommandBuffer cmd, VkBuffer indirectCommandBuffer)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline);
    // set dynamic state
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    // push constants
    vkCmdPushConstants(cmd, GraphicsPipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(VoxelPushConstants), &PushConstants);
    // bind descriptors
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipelineLayout, 1, 1, &GraphicsDescriptorSet, 0, nullptr);

    vkCmdDrawMeshTasksIndirectEXT(cmd, indirectCommandBuffer, 0, 1, 0);
}

void VoxelRenderingIndirectPass::SetGridShellSizes(const glm::uvec3& chunkSize, const glm::uvec3& shellSize)
{
    PushConstants.chunkSize = chunkSize;
    PushConstants.shellSize = shellSize;
}

void VoxelRenderingIndirectPass::SetDepthPyramidBinding(VulkanEngine* engine, VkImageView depthPyramidView, VkSampler depthPyramidSampler)
{
    DescriptorWriter writer;
    writer.writeImage(0, depthPyramidView, depthPyramidSampler, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.updateSet(engine->device, ComputeDescriptorSet);
}

void VoxelRenderingIndirectPass::SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& chunkDrawDataBufferAddress, const VkDeviceAddress& drawChunkCountBufferAddress)
{
    PushConstants.chunkMetadataBufferAddress = chunkMetadataBufferAddress;
    PushConstants.chunkDrawDataBufferAddress = chunkDrawDataBufferAddress;
    PushConstants.drawChunkCountBufferAddress = drawChunkCountBufferAddress;
}

void VoxelRenderingIndirectPass::SetVoxelSize(const glm::vec3& voxelSize)
{
    PushConstants.voxelSize = voxelSize;
}

void VoxelRenderingIndirectPass::SetNumChunks(uint32_t numChunks)
{
    PushConstants.numChunks = numChunks;
}

void VoxelRenderingIndirectPass::SetCameraZNear(float zNear)
{
    PushConstants.zNear = zNear;
}

void VoxelRenderingIndirectPass::SetDepthPyramidSizes(uint32_t depthPyramidWidth, uint32_t depthPyramidHeight)
{
    PushConstants.depthPyramidWidth = depthPyramidWidth;
    PushConstants.depthPyramidHeight = depthPyramidHeight;
}

void VoxelRenderingIndirectPass::SetColorPaletteBinding(VulkanEngine* engine, VkBuffer colorPaletteBuffer, size_t bufferSize)
{
    DescriptorWriter writer;
    writer.writeBuffer(0, colorPaletteBuffer, bufferSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.updateSet(engine->device, GraphicsDescriptorSet);
}

void VoxelRenderingIndirectPass::ClearResources(VulkanEngine* engine)
{
    vkDestroyDescriptorSetLayout(engine->device, GraphicsDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine->device, ComputeDescriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(engine->device, GraphicsPipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, GraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(engine->device, ComputePipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, ComputePipeline, nullptr);
}
