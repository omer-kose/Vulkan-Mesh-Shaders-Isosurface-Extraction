#include "CTheadScene.h"

#include <Core/vk_engine.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>

#include <fstream>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

void CTheadScene::load(VulkanEngine* engine)
{
    pEngine = engine;

    glm::uvec3 gridSize;
    std::tie(voxelBuffer, gridSize) = loadCTheadData();

    mcSettings.gridSize = glm::uvec3(gridSize.x, gridSize.z, gridSize.y);
    mcSettings.shellSize = mcSettings.gridSize;
    mcSettings.isoValue = 0.5f;

    MarchingCubesPass::SetVoxelBufferDeviceAddress(pEngine->getBufferDeviceAddress(voxelBuffer.buffer));
    MarchingCubesPass::SetGridCornerPositions(glm::vec3(-0.5f), glm::vec3(0.5f));
    MarchingCubesPass::UpdateMCSettings(mcSettings);;

    // Set the camera
    mainCamera = Camera(glm::vec3(-2.0f, 0.0f, 2.0f), 0.0f, -45.0f);
    mainCamera.setSpeed(0.02f);

    // Set attachment clear color
    pEngine->setColorAttachmentClearColor(VkClearValue{0.6f, 0.9f, 1.0f, 1.0f});

    // Set Grid Plane height
    CircleGridPlanePass::SetPlaneHeight(-0.1f);
}

void CTheadScene::processSDLEvents(SDL_Event& e)
{
    mainCamera.processSDLEvent(e);
}

void CTheadScene::handleUI()
{
    ImGui::Begin("Marching Cubes Parameters");
    ImGui::SliderFloat("Iso Value", &mcSettings.isoValue, 0.0f, 1.0f);
    ImGui::End();
}

void CTheadScene::update()
{
    mainCamera.update();

    sceneData.view = mainCamera.getViewMatrix();

    VkExtent2D windowExtent = pEngine->getWindowExtent();
    // camera projection
    sceneData.proj = glm::perspectiveRH_ZO(glm::radians(45.f), (float)windowExtent.width / (float)windowExtent.height, 0.1f, 10000.f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(0.1f);
    sceneData.sunlightColor = glm::vec4(1.0f);
    glm::vec3 directionalLightDir = glm::normalize(glm::vec3(0.0f, -1.0f, -0.5f));
    sceneData.sunlightDirection = glm::vec4(directionalLightDir, 1.0f);

    // Update the MC params (cheap operation but could be checked if there is any change)
    MarchingCubesPass::UpdateMCSettings(mcSettings);
}

void CTheadScene::drawFrame(VkCommandBuffer cmd)
{
    MarchingCubesPass::Execute(pEngine, cmd);
    CircleGridPlanePass::Execute(pEngine, cmd);
}

CTheadScene::~CTheadScene()
{
    pEngine->destroyBuffer(voxelBuffer);
}

std::pair<AllocatedBuffer, glm::uvec3> CTheadScene::loadCTheadData() const
{
    /*
         Load CT Head data. It is given in bytes. Format is 16-bit integers where two consecutive bytes make up one binary integer.
         The loading procedure is:
         1- Read in the bytes
         2- Dispatch a compute shader to convert: unsigned short -> float where
    */
    // Open the file with the cursor at the end
    std::ifstream file("../../assets/CThead/CThead.bytes", std::ios::ate | std::ios::binary);

    if(!file.is_open())
    {
        fmt::println("Error when loading CThead data");
        return {};
    }

    // As the cursor is already at the end, we can directly asses the byte size of the file
    size_t fileSize = (size_t)file.tellg();

    // Store the shader code
    std::vector<char> buffer(fileSize);

    // Put the cursor at the beginning
    file.seekg(0);

    // Load the entire file into the buffer (read() reads the file byte by byte)
    file.read(buffer.data(), fileSize);

    // We are done with the file
    file.close();

    glm::uvec3 gridSize = glm::uvec3(256, 256, 113); // hardcoded by data

    // Create the temp compute shader pipeline
    struct VolumeDataConverterPushConstants
    {
        glm::uvec3 gridSize;
        VkDeviceAddress sourceBufferAddress;
        VkDeviceAddress voxelBufferAddress;
    };

    VolumeDataConverterPushConstants converterPC;
    converterPC.gridSize = gridSize;
    size_t voxelBufferSize = converterPC.gridSize.x * converterPC.gridSize.y * converterPC.gridSize.z * sizeof(float);

    // Load the source data into GPU and fetch the address
    AllocatedBuffer sourceBuffer = pEngine->createAndUploadGPUBuffer(fileSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, buffer.data());
    converterPC.sourceBufferAddress = pEngine->getBufferDeviceAddress(sourceBuffer.buffer);

    // Create the voxel buffer that will be written on by the compute kernel and fetch the address
    AllocatedBuffer voxelBuffer = pEngine->createBuffer(voxelBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    converterPC.voxelBufferAddress = pEngine->getBufferDeviceAddress(voxelBuffer.buffer);

    // Create the compute pipeline
    VkPipelineLayoutCreateInfo converterPipelineLayoutInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr };
    VkPushConstantRange converterPCRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(VolumeDataConverterPushConstants) };
    converterPipelineLayoutInfo.pushConstantRangeCount = 1;
    converterPipelineLayoutInfo.pPushConstantRanges = &converterPCRange;

    VkPipelineLayout converterPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(pEngine->device, &converterPipelineLayoutInfo, nullptr, &converterPipelineLayout));
    VkShaderModule converterComputeShader;
    if(!vkutil::loadShaderModule(pEngine->device, "../../shaders/glsl/volume_data_convert/volume_data_convert_comp.spv", &converterComputeShader))
    {
        fmt::println("Volume Data Converter Compute Shader could not be loaded!");
    }

    VkPipelineShaderStageCreateInfo converterShaderStageInfo = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, converterComputeShader);
    VkComputePipelineCreateInfo converterPipelineInfo = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr };
    converterPipelineInfo.layout = converterPipelineLayout;
    converterPipelineInfo.stage = converterShaderStageInfo;

    VkPipeline converterPipeline;
    VK_CHECK(vkCreateComputePipelines(pEngine->device, VK_NULL_HANDLE, 1, &converterPipelineInfo, nullptr, &converterPipeline));

    auto ceilDiv = [](unsigned int x, unsigned int y) { return (x + y - 1) / y; };
    // Immediate dispatch the converter kernel
    pEngine->immediateSubmit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, converterPipeline);
        vkCmdPushConstants(cmd, converterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VolumeDataConverterPushConstants), &converterPC);

        vkCmdDispatch(cmd, ceilDiv(converterPC.gridSize.x, 8u), ceilDiv(converterPC.gridSize.y, 8u), ceilDiv(converterPC.gridSize.z, 8u));
    });

    // Delete the temporary resources
    vkDestroyPipelineLayout(pEngine->device, converterPipelineLayout, nullptr);
    vkDestroyPipeline(pEngine->device, converterPipeline, nullptr);
    vkDestroyShaderModule(pEngine->device, converterComputeShader, nullptr);
    pEngine->destroyBuffer(sourceBuffer);

    return { voxelBuffer, gridSize };
}
