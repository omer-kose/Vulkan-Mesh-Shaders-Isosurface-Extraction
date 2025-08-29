#include "ChunkVisualizationPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline ChunkVisualizationPass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout ChunkVisualizationPass::PipelineLayout = VK_NULL_HANDLE;
ChunkVisualizationPass::ChunkVisPushConstants ChunkVisualizationPass::PushConstants = {};

void ChunkVisualizationPass::Init(VulkanEngine* engine)
{
    // Init the pipeline
    // Load the shaders
    VkShaderModule vertexShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/chunk_visualization/chunk_visualization_vert.spv", &vertexShader))
    {
        fmt::println("Error when building chunk visualization vertex shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/chunk_visualization/chunk_visualization_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building chunk visualization fragment shader");
    }

    // Push Constant (MC Settings are dynamic and updated via UpdateMCSettings function if needed (needs to be updated at least once of course))
    VkPushConstantRange pcRange{ .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(ChunkVisPushConstants) };

    // Set descriptor sets (only scene descriptor set is used for camera info)
    VkDescriptorSetLayout layouts[] = { engine->getSceneDescriptorLayout() };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = layouts;

    VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    // Build the pipeline
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.setShaders(vertexShader, fragmentShader);
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_GREATER);

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);

    pipelineBuilder.pipelineLayout = PipelineLayout;
    Pipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, vertexShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
}

void ChunkVisualizationPass::Execute(VulkanEngine* engine, VkCommandBuffer cmd, size_t numChunks, float lineWidth)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    // set dynamic state
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    vkCmdSetLineWidth(cmd, lineWidth);
    // push constants
    vkCmdPushConstants(cmd, PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ChunkVisPushConstants), &PushConstants);
    // bind scene descriptor set
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdDraw(cmd, 24, numChunks, 0, 0); // 12 edges for a cube thus 24 vertices
}

void ChunkVisualizationPass::SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& activeChunkIndicesBuffer)
{
    PushConstants.chunkMetadataBufferAddress = chunkMetadataBufferAddress;
    PushConstants.activeChunkIndicesBuffer = activeChunkIndicesBuffer;
}

void ChunkVisualizationPass::SetNumActiveChunks(uint32_t numActiveChunks)
{
	PushConstants.numActiveChunks = numActiveChunks;
}

void ChunkVisualizationPass::ClearResources(VulkanEngine* engine)
{
	vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
	vkDestroyPipeline(engine->device, Pipeline, nullptr);
}
