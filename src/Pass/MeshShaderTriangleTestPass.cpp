#include "MeshShaderTriangleTestPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline MeshShaderTriangleTestPass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout MeshShaderTriangleTestPass::PipelineLayout = VK_NULL_HANDLE;

void MeshShaderTriangleTestPass::Init(VulkanEngine* engine)
{
    // Load the shaders
    VkShaderModule meshShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/mesh_shader_triangle_test/mesh_shader_triangle_test_mesh.spv", &meshShader))
    {
        fmt::println("Error when building the basic triangle test mesh shader");
    }

    VkShaderModule fragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/mesh_shader_triangle_test/mesh_shader_triangle_test_frag.spv", &fragmentShader))
    {
        fmt::println("Error when building the basic triangle test fragment shader");
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.setLayoutCount = 0;

    VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    // Build the pipeline
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.pushShaderStage(meshShader, VK_SHADER_STAGE_MESH_BIT_EXT);
    pipelineBuilder.pushShaderStage(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.disableDepthTest();

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);

    pipelineBuilder.pipelineLayout = PipelineLayout;
    // Opaque Pipeline
    Pipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, meshShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
}

void MeshShaderTriangleTestPass::Execute(VulkanEngine* engine, VkCommandBuffer& cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    engine->setViewport(cmd);
    engine->setScissor(cmd);

    vkCmdDrawMeshTasksEXT(cmd, 1, 1, 1);
}

void MeshShaderTriangleTestPass::Update()
{
}

void MeshShaderTriangleTestPass::ClearResources(VulkanEngine* engine)
{
    vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, Pipeline, nullptr);
}
