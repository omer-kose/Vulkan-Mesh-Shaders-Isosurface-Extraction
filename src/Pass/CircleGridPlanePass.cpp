#include "CircleGridPlanePass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline CircleGridPlanePass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout CircleGridPlanePass::PipelineLayout = VK_NULL_HANDLE;
CircleGridPlanePass::GridPlanePushConstants CircleGridPlanePass::PushConstants = {};

void CircleGridPlanePass::Init(VulkanEngine* engine)
{
    // Load the shaders
    VkShaderModule vertexShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/circle_grid_plane/circle_grid_plane_vert.spv", &vertexShader))
    {
        fmt::println("Error when building the grid plane vertex shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/circle_grid_plane/circle_grid_plane_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building the grid plane fragment shader");
    }

    VkPushConstantRange pcRange = { .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(GridPlanePushConstants) };

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
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.enableBlendingAlphaBlend();
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);

    pipelineBuilder.pipelineLayout = PipelineLayout;
    Pipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, vertexShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
}

void CircleGridPlanePass::Execute(VulkanEngine* engine, VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    vkCmdPushConstants(cmd, PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GridPlanePushConstants), &PushConstants);
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &sceneDescriptorSet, 0, 0);

    vkCmdDraw(cmd, 6, 1, 0, 0);
}

void CircleGridPlanePass::Update()
{
}

void CircleGridPlanePass::ClearResources(VulkanEngine* engine)
{
    vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, Pipeline, nullptr);
}

void CircleGridPlanePass::SetPlaneHeight(float height)
{
    PushConstants.planeHeight = height;
}
