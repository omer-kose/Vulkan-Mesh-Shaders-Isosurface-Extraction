#include "GLTFMetallicPass.h"

#include <Core/vk_engine.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_initializers.h>

// Define the static members
VkPipeline GLTFMetallicPass::OpaquePipeline = VK_NULL_HANDLE;
VkPipeline GLTFMetallicPass::TransparentPipeline = VK_NULL_HANDLE;
VkPipelineLayout GLTFMetallicPass::PipelineLayout = VK_NULL_HANDLE;

void GLTFMetallicPass::Init(VulkanEngine* engine)
{
    // Load the shaders
    VkShaderModule meshVertexShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/gltf_metallic/mesh_vert.spv", &meshVertexShader))
    {
        fmt::println("Error when building the mesh vertex shader");
    }

    VkShaderModule meshFragmentShader;
    if(!vkutil::loadShaderModule(engine->device, "../../shaders/glsl/gltf_metallic/mesh_frag.spv", &meshFragmentShader))
    {
        fmt::println("Error when building the mesh fragment shader");
    }

    // Set push constant range
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(GPUDrawPushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    // Set descriptor sets
    // Material set (set 1)
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    VkDescriptorSetLayout materialLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    // 2 sets: 0 -> Scene Descriptor Set, 1 -> Material Descriptor Set
    VkDescriptorSetLayout layouts[] = { engine->getSceneDescriptorLayout(), materialLayout};

    // Mesh pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;

    VK_CHECK(vkCreatePipelineLayout(engine->device, &pipelineLayoutInfo, nullptr, &PipelineLayout));

    // Build the pipeline
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.setShaders(meshVertexShader, meshFragmentShader);
    pipelineBuilder.setInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.setPolygonMode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.setCullMode(VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    pipelineBuilder.setMultiSamplingNone();
    pipelineBuilder.disableBlending();
    pipelineBuilder.enableDepthTest(true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // Render format
    pipelineBuilder.setColorAttachmentFormat(engine->drawImage.imageFormat);
    pipelineBuilder.setDepthFormat(engine->depthImage.imageFormat);

    pipelineBuilder.pipelineLayout = PipelineLayout;
    // Opaque Pipeline
    OpaquePipeline = pipelineBuilder.buildPipeline(engine->device);

    // Transparent variant
    pipelineBuilder.enableBlendingAdditive();
    pipelineBuilder.enableDepthTest(false, VK_COMPARE_OP_LESS_OR_EQUAL);
    TransparentPipeline = pipelineBuilder.buildPipeline(engine->device);

    // ShaderModules are not needed anymore
    vkDestroyShaderModule(engine->device, meshVertexShader, nullptr);
    vkDestroyShaderModule(engine->device, meshFragmentShader, nullptr);
    // Descriptor Set Layout is not needed as Material descriptors will be created while getting instanced.
    vkDestroyDescriptorSetLayout(engine->device, materialLayout, nullptr);
}

void GLTFMetallicPass::Execute(VulkanEngine* engine, VkCommandBuffer& cmd)
{
    const DrawContext* ctx = engine->getDrawContext();
    std::vector<uint32_t> opaqueDraws;
    opaqueDraws.reserve(ctx->opaqueGLTFSurfaces.size());

    for(uint32_t i = 0; i < ctx->opaqueGLTFSurfaces.size(); ++i)
    {
        opaqueDraws.push_back(i);
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaqueDraws.begin(), opaqueDraws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = ctx->opaqueGLTFSurfaces[iA];
        const RenderObject& B = ctx->opaqueGLTFSurfaces[iB];
        if(A.materialInstance == B.materialInstance)
        {
            return A.indexBuffer < B.indexBuffer;
        }
        else
        {
            return A.materialInstance < B.materialInstance;
        }
    });

    // Keep track of states to avoid unnecessary rebindings
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject& robj, VkPipeline pipeline) {
        if(robj.materialInstance != lastMaterial)
        {
            lastMaterial = robj.materialInstance;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1, &sceneDescriptorSet, 0, nullptr);

            // Set dynamic viewport and scissor again in case of an override (all of the material pipelines use dynamic states so setting them once after a bind is actually enough)
            engine->setViewport(cmd);
            engine->setScissor(cmd);

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 1, 1, &robj.materialInstance->materialSet, 0, nullptr);
        }

        GPUDrawPushConstants pushConstants;
        pushConstants.vertexBufferAddress = robj.vertexBufferAddress;
        pushConstants.worldMatrix = robj.transform;
        vkCmdPushConstants(cmd, PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

        if(lastIndexBuffer != robj.indexBuffer)
        {
            lastIndexBuffer = robj.indexBuffer;
            vkCmdBindIndexBuffer(cmd, robj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        vkCmdDrawIndexed(cmd, robj.indexCount, 1, robj.firstIndex, 0, 0);
    };

    for(uint32_t idx : opaqueDraws)
    {
        draw(ctx->opaqueGLTFSurfaces[idx], OpaquePipeline);
    }

    for(const RenderObject& robj : ctx->transparentGLTFSurfaces)
    {
        draw(robj, TransparentPipeline);
    }
}

void GLTFMetallicPass::Update()
{
}

void GLTFMetallicPass::ClearResources(VulkanEngine* engine)
{
    vkDestroyPipelineLayout(engine->device, PipelineLayout, nullptr);

    vkDestroyPipeline(engine->device, OpaquePipeline, nullptr);
    vkDestroyPipeline(engine->device, TransparentPipeline, nullptr);
}
