#include "MarchingCubesPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline MarchingCubesPass::Pipeline = VK_NULL_HANDLE;
VkPipelineLayout MarchingCubesPass::PipelineLayout = VK_NULL_HANDLE;
VkDescriptorSet MarchingCubesPass::MCDescriptorSet = VK_NULL_HANDLE;
AllocatedBuffer MarchingCubesPass::MCLookupTableBuffer = {};
AllocatedBuffer MarchingCubesPass::MCSettingsBuffer = {};
MarchingCubesPass::MCSettings MarchingCubesPass::Settings = {};

template<typename T>
T ceilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

void MarchingCubesPass::Init(VulkanEngine* engine, const MCSettings& mcSettings_in)
{
    // Init the resources
    size_t lookupTableSize = sizeof(MarchingCubesLookupTable);
	MCLookupTableBuffer = engine->createAndUploadGPUBuffer(lookupTableSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, (void*)MarchingCubesLookupTable);

    size_t mcSettingsSize = sizeof(MCSettings);
    Settings = mcSettings_in;
    MCSettingsBuffer = engine->createAndUploadGPUBuffer(mcSettingsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY, (void*)&Settings);

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

    // Set descriptor sets
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    VkDescriptorSetLayout mcSetLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT);
    // Allocate the descriptor set and update
    MCDescriptorSet = engine->globalDescriptorAllocator.allocate(engine->device, mcSetLayout);
    DescriptorWriter writer;
    writer.writeBuffer(0, MCLookupTableBuffer.buffer, lookupTableSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.writeBuffer(1, MCSettingsBuffer.buffer, mcSettingsSize, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.updateSet(engine->device, MCDescriptorSet);

    // 2 sets: 0 -> Scene Descriptor Set, 1 -> Pass Specific Descriptor Set
    VkDescriptorSetLayout layouts[] = { engine->getSceneDescriptorLayout(), mcSetLayout };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;

    VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    // Build the pipeline
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.pushShaderStage(taskShader, VK_SHADER_STAGE_TASK_BIT_EXT);
    pipelineBuilder.pushShaderStage(meshShader, VK_SHADER_STAGE_MESH_BIT_EXT);
    pipelineBuilder.pushShaderStage(fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);

    pipelineBuilder.pipelineLayout = PipelineLayout;
    Pipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, taskShader, nullptr);
    vkDestroyShaderModule(engine->device, meshShader, nullptr);
    vkDestroyShaderModule(engine->device, fragmentShader, nullptr);
    vkDestroyDescriptorSetLayout(engine->device, mcSetLayout, nullptr);
}

void MarchingCubesPass::Execute(VulkanEngine* engine, VkCommandBuffer& cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline);
    // set dynamic state
    engine->setViewport(cmd);
    engine->setScissor(cmd);
    // bind descriptors
    VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 1, 1, &MCDescriptorSet, 0, nullptr);

    vkCmdDrawMeshTasksEXT(cmd, ceilDiv(Settings.gridSize.x, 4u) * ceilDiv(Settings.gridSize.y, 4u) * ceilDiv(Settings.gridSize.z, 4u), 1, 1);
}

void MarchingCubesPass::Update()
{
}

void MarchingCubesPass::ClearResources(VulkanEngine* engine)
{
    vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);
    vkDestroyPipeline(engine->device, Pipeline, nullptr);
    engine->destroyBuffer(MCLookupTableBuffer);
    engine->destroyBuffer(MCSettingsBuffer);
}
