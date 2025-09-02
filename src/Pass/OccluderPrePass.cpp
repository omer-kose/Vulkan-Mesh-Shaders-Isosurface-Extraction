#include "OccluderPrePass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline OccluderPrePass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout OccluderPrePass::PipelineLayout = VK_NULL_HANDLE;
OccluderPrePass::OccluderPushConstants OccluderPrePass::PushConstants = {};

void OccluderPrePass::Init(VulkanEngine* engine)
{
    // Init the pipeline
    // Load the shaders
    VkShaderModule meshShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/occluder_prepass/occluder_prepass_mesh.spv", &meshShader))
    {
        fmt::println("Error when building occluder prepass mesh shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/occluder_prepass/occluder_prepass_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building occluder prepass fragment shader");
    }

    // Push Constant (MC Settings are dynamic and updated via UpdateMCSettings function if needed (needs to be updated at least once of course))
    VkPushConstantRange pcRange{ .stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(OccluderPushConstants) };

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
    pipelineBuilder.pushShaderStage(meshShader, VK_SHADER_STAGE_MESH_BIT_EXT);
    pipelineBuilder.pushShaderStage(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
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
    vkDestroyShaderModule(engine->device, meshShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
}

void OccluderPrePass::Execute(VulkanEngine* engine, VkCommandBuffer cmd, VkBuffer indirectCommandBuffer)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    // set dynamic state
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    // push constants
    vkCmdPushConstants(cmd, PipelineLayout, VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(OccluderPushConstants), &PushConstants);
    // bind scene descriptor set
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdDrawMeshTasksIndirectEXT(cmd, indirectCommandBuffer, 0, 1, 0);
}

void OccluderPrePass::SetChunkBufferAddresses(const VkDeviceAddress& chunkMetadataBufferAddress, const VkDeviceAddress& chunkDrawDataBufferAddress)
{
    PushConstants.chunkMetadataBufferAddress = chunkMetadataBufferAddress;
    PushConstants.chunkDrawDataBufferAddress = chunkDrawDataBufferAddress;
}

void OccluderPrePass::SetChunkSize(const glm::uvec3& chunkSize)
{
    PushConstants.chunkSize = chunkSize;
}

void OccluderPrePass::ClearResources(VulkanEngine* engine)
{
	vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
	vkDestroyPipeline(engine->device, Pipeline, nullptr);
}
