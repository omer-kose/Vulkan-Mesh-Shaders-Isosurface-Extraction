#include "VoxelRenderingScene.h"

#include <Core/vk_engine.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_barriers.h>

#include <fstream>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <execution>
#include <random>

void VoxelRenderingScene::load(VulkanEngine* engine)
{
    pEngine = engine;

    chunkSize = glm::uvec3(32, 32, 32);
    blockSize = 4;
    blocksPerChunk = (chunkSize.x * chunkSize.y * chunkSize.z) / (blockSize * blockSize * blockSize);
    gridLowerCornerPos = glm::vec3(-0.5f);
    gridUpperCornerPos = glm::vec3(0.5f);

    // organNames = { "CThead", "Kidney" };

    selectedModelID = 0;
    loadData(selectedModelID);

    // Set the camera
    mainCamera = Camera(glm::vec3(-2.0f, 0.0f, 2.0f), 0.0f, -45.0f);
    mainCamera.setSpeed(0.02f);

    // Set attachment clear color
    pEngine->setColorAttachmentClearColor(VkClearValue{ 0.6f, 0.9f, 1.0f, 1.0f });

    VoxelRenderingIndirectPass::SetDepthPyramidBinding(pEngine, HZBDownSamplePass::GetDepthPyramidImageView(), HZBDownSamplePass::GetDepthPyramidSampler());
    VoxelRenderingIndirectPass::SetDepthPyramidSizes(HZBDownSamplePass::GetDepthPyramidWidth(), HZBDownSamplePass::GetDepthPyramidHeight());
}

void VoxelRenderingScene::processSDLEvents(SDL_Event& e)
{
    mainCamera.processSDLEvent(e);
}

void VoxelRenderingScene::handleUI()
{
    ImGui::Begin("Marching Cubes Parameters");

    /*
        // Model Selection
        if(ImGui::BeginCombo("Model Selection", organNames[selectedOrganID].c_str()))
        {
            for(int i = 0; i < organNames.size(); i++)
            {
                const bool isSelected = (selectedOrganID == i);
                if(ImGui::Selectable(organNames[i].c_str(), isSelected))
                {
                    if(selectedOrganID != i)
                    {
                        selectedOrganID = i;
                        loadData(selectedOrganID);
                    }
                }

                // Set the initial focus when opening the combo
                if(isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    */
    ImGui::Checkbox("Show Chunks", &showChunks);
    ImGui::End();
}

void VoxelRenderingScene::update()
{
    mainCamera.update();

    sceneData.view = mainCamera.getViewMatrix();
    constexpr float fov = glm::radians(45.0f);
    constexpr float zNear = 0.01f;
    constexpr float zFar = 10000.f;

    VkExtent2D windowExtent = pEngine->getWindowExtent();

    const float f = 1.0f / tanf(fov / 2.0f);;
    const float aspectRatio = float(windowExtent.width) / float(windowExtent.height);
    sceneData.proj = glm::mat4(
        f / aspectRatio, 0.0f, 0.0f, 0.0f,
        0.0f, -f, 0.0f, 0.0f, // inverting y inplace for Vulkan's convention
        0.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, zNear, 0.0f
    );

    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(0.1f);
    sceneData.sunlightColor = glm::vec4(1.0f);
    glm::vec3 directionalLightDir = glm::normalize(glm::vec3(0.0f, -1.0f, -0.5f));
    sceneData.sunlightDirection = glm::vec4(directionalLightDir, 1.0f);

    // Update the MC params (cheap operation but could be checked if there is any change)
    VoxelRenderingIndirectPass::SetCameraZNear(zNear);
    VoxelRenderingIndirectPass::SetCameraPos(mainCamera.position);
    ChunkVisualizationPass::SetInputIsovalue(0.5f); // Setting this to any value as there is no isovalue in Voxel Rendering
}

void VoxelRenderingScene::drawFrame(VkCommandBuffer cmd)
{
    if(showChunks)
    {
        ChunkVisualizationPass::Execute(pEngine, cmd, chunkedVolumeData->getNumChunksFlat(), 3.0f);
    }

    VoxelRenderingIndirectPass::ExecuteGraphicsPass(pEngine, cmd, drawChunkCountBuffer.buffer);
}

void VoxelRenderingScene::performPreRenderPassOps(VkCommandBuffer cmd)
{
    // Clear draw count back to 0 
    vkCmdFillBuffer(cmd, drawChunkCountBuffer.buffer, 0, 4, 0);
    vkCmdFillBuffer(cmd, drawChunkCountBuffer.buffer, 4, 8, 1); // set y and z group numbers to 1
    VkBufferMemoryBarrier2 fillBarrier = vkutil::bufferBarrier(drawChunkCountBuffer.buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    vkutil::pipelineBarrier(cmd, 0, 1, &fillBarrier, 0, nullptr);

    // Compute Culling and Preparing Indirect Commands Pass
    size_t totalTasks = chunkedVolumeData->getNumChunksFlat() * blocksPerChunk;
    VoxelRenderingIndirectPass::ExecuteComputePass(pEngine, cmd, totalTasks);
    // Synchronize and Protect drawData and drawCount buffers before indirect dispatch
    VkBufferMemoryBarrier2 cullBarriers[] = {
        vkutil::bufferBarrier(chunkDrawDataBuffer.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,  VK_ACCESS_SHADER_READ_BIT),
        vkutil::bufferBarrier(drawChunkCountBuffer.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
    };

    vkutil::pipelineBarrier(cmd, 0, 2, cullBarriers, 0, nullptr);
}

void VoxelRenderingScene::performPostRenderPassOps(VkCommandBuffer cmd)
{
    HZBDownSamplePass::Execute(pEngine, cmd);
}

VoxelRenderingScene::~VoxelRenderingScene()
{
    clearBuffers();
}

std::vector<uint8_t> VoxelRenderingScene::createRandomVoxelData(const glm::uvec3& gridSize)
{
    size_t numVoxels = gridSize.x * gridSize.y * gridSize.z;
    std::vector<uint8_t> gridData(numVoxels);
    std::vector<uint32_t> indices(numVoxels);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](uint32_t i){
        gridData[i] = rand() % 2;

    });
    return gridData;
}

void VoxelRenderingScene::createChunkVisualizationBuffer(const std::vector<VolumeChunk>& chunks)
{
    struct ChunkVisInformation
    {
        glm::vec3 lowerCornerPos;
        glm::vec3 upperCornerPos;
        float minIsoValue;
        float maxIsoValue;
    };

    size_t numChunks = chunks.size();
    size_t stagingBufferSize = numChunks * sizeof(ChunkVisInformation);
    std::vector<ChunkVisInformation> chunkVisInfo(numChunks);
    // Fill in the chunk visualization info
    for(size_t i = 0; i < numChunks; ++i)
    {
        const VolumeChunk& chunk = chunks[i];
        chunkVisInfo[i].lowerCornerPos = chunk.lowerCornerPos;
        chunkVisInfo[i].upperCornerPos = chunk.upperCornerPos;
        chunkVisInfo[i].minIsoValue = chunk.minIsoValue;
        chunkVisInfo[i].maxIsoValue = chunk.maxIsoValue;
    }

    // Create the Chunk Visualization GPU buffer from the staging buffer
    chunkVisualizationBuffer = pEngine->createAndUploadGPUBuffer(stagingBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, chunkVisInfo.data());
    chunkVisualizationBufferAddress = pEngine->getBufferDeviceAddress(chunkVisualizationBuffer.buffer);
}

void VoxelRenderingScene::loadData(uint32_t modelID)
{
    // Make sure that Vulkan is done working with the buffer (not the best way but in this case this scene does not do anything else other than rendering a geometry)
    vkDeviceWaitIdle(pEngine->device);
    clearBuffers();
    std::vector<uint8_t> gridData; glm::uvec3 gridSize;

    switch(modelID)
    {
    default:
        // fmt::println("No existing model is selected!");
        break;
    }

    // TODO: Testing random data
    gridSize = glm::uvec3(256);
    gridData = createRandomVoxelData(gridSize);

    // Create the chunked version of the grid
    chunkedVolumeData = std::make_unique<ChunkedVolumeData<uint8_t>>(pEngine, gridData, gridSize, chunkSize, gridLowerCornerPos, gridUpperCornerPos, false);
    gridData.clear();

    // Allocate the chunk buffer on GPU and load the whole data at the beginning only once
    size_t numChunks = chunkedVolumeData->getNumChunksFlat();
    size_t voxelChunksBufferSize = numChunks * chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(uint8_t);
    voxelChunksBuffer = pEngine->uploadStagingBuffer(chunkedVolumeData->getStagingBuffer(), voxelChunksBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    voxelChunksBufferBaseAddress = pEngine->getBufferDeviceAddress(voxelChunksBuffer.buffer);

    // Once the data is loaded staging buffer is no longer needed
    chunkedVolumeData->destroyStagingBuffer();

    // Allocate Indirect Buffers
    const std::vector<VolumeChunk>& chunks = chunkedVolumeData->getChunks();
    std::vector<VoxelRenderingIndirectPass::ChunkMetadata> chunkMetadata(numChunks);
    {
        std::vector<size_t> indices(numChunks);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            chunkMetadata[i].lowerCornerPos = chunks[i].lowerCornerPos;
            chunkMetadata[i].upperCornerPos = chunks[i].upperCornerPos;
            chunkMetadata[i].voxelBufferDeviceAddress = voxelChunksBufferBaseAddress + chunks[i].stagingBufferOffset;
        });
    }
    chunkMetadataBuffer = pEngine->createAndUploadGPUBuffer(numChunks * sizeof(VoxelRenderingIndirectPass::ChunkMetadata), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, chunkMetadata.data());

    size_t maxNumTaskInvocations = numChunks * ((chunkSize.x * chunkSize.y * chunkSize.z) / (blockSize * blockSize * blockSize));
    chunkDrawDataBuffer = pEngine->createBuffer(maxNumTaskInvocations * sizeof(VoxelRenderingIndirectPass::ChunkDrawData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    drawChunkCountBuffer = pEngine->createBuffer(3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // Set Voxel Rendering Params
    VoxelRenderingIndirectPass::SetGridShellSizes(chunkSize, chunkedVolumeData->getShellSize());
    VoxelRenderingIndirectPass::SetChunkBufferAddresses(pEngine->getBufferDeviceAddress(chunkMetadataBuffer.buffer), pEngine->getBufferDeviceAddress(chunkDrawDataBuffer.buffer), pEngine->getBufferDeviceAddress(drawChunkCountBuffer.buffer));
    // TODO: Set remaining push constants!!
    VoxelRenderingIndirectPass::SetNumChunks(chunkedVolumeData->getNumChunksFlat());
    glm::vec3 voxelSize = (gridUpperCornerPos - gridLowerCornerPos) / glm::vec3(gridSize - 1u);
    VoxelRenderingIndirectPass::SetVoxelSize(voxelSize);

    // Prepare Chunk Visualization
    createChunkVisualizationBuffer(chunkedVolumeData->getChunks());
    ChunkVisualizationPass::SetChunkBufferDeviceAddress(chunkVisualizationBufferAddress);
    ChunkVisualizationPass::SetInputIsovalue(0.5f); // a random value bigger than 0
}

void VoxelRenderingScene::clearBuffers()
{
    pEngine->destroyBuffer(voxelChunksBuffer);
    pEngine->destroyBuffer(chunkMetadataBuffer);
    pEngine->destroyBuffer(chunkDrawDataBuffer);
    pEngine->destroyBuffer(drawChunkCountBuffer);
    pEngine->destroyBuffer(chunkVisualizationBuffer);
}
