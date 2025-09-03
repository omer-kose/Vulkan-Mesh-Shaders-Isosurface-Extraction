#include "VoxelRenderingScene.h"

#include <Core/vk_engine.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_barriers.h>

#define OGT_VOX_IMPLEMENTATION
#include <Data/ogt_vox.h>

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

    modelNames = { "biome", "monument", "teapot" };

    selectedModelID = 0;
    loadData(selectedModelID);

    // Set the camera
    mainCamera = Camera(glm::vec3(0.0f, 0.0f, 2.0f), 0.0f, 0.0f);
    mainCamera.setSpeed(2.0f);

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
    ImGui::Begin("Voxel Renderer Parameters");

    
        // Model Selection
        if(ImGui::BeginCombo("Model Selection", modelNames[selectedModelID].c_str()))
        {
            for(int i = 0; i < modelNames.size(); i++)
            {
                const bool isSelected = (selectedModelID == i);
                if(ImGui::Selectable(modelNames[i].c_str(), isSelected))
                {
                    if(selectedModelID != i)
                    {
                        selectedModelID = i;
                        loadData(selectedModelID);
                    }
                }

                // Set the initial focus when opening the combo
                if(isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    
    ImGui::Checkbox("Show Chunks", &showChunks);
    ImGui::End();
}

void VoxelRenderingScene::update(float dt)
{
    mainCamera.update(dt);

    sceneData.cameraPos = mainCamera.position;
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
}

void VoxelRenderingScene::drawFrame(VkCommandBuffer cmd)
{
    if(showChunks)
    {
        ChunkVisualizationPass::Execute(pEngine, cmd, numActiveChunks, 3.0f);
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
    sceneData.prevViewProj = sceneData.viewproj;
}

VoxelRenderingScene::~VoxelRenderingScene()
{
    clearBuffers();
}

void VoxelRenderingScene::fillRandomVoxelData(std::vector<uint8_t>& grid, float fillProbability, int seed)
{
    size_t numVoxels = grid.size();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<uint8_t> gridData(numVoxels);
    std::for_each(std::execution::par, grid.begin(), grid.end(), [&](uint8_t& voxelVal){
        voxelVal = (dist(rng) < fillProbability) ? 1 : 0;
    });
}

float hash3D(int x, int y, int z)
{
    int h = x * 374761393 + y * 668265263 + z * 73856093;
    h = (h ^ (h >> 13)) * 1274126177;
    return ((h & 0x7fffffff) / float(0x7fffffff));
}

float noise3D(float x, float y, float z, float scale = 0.05f)
{
    int xi = int(std::floor(x * scale));
    int yi = int(std::floor(y * scale));
    int zi = int(std::floor(z * scale));
    return hash3D(xi, yi, zi);
}

void VoxelRenderingScene::generateVoxelScene(std::vector<uint8_t>& grid, int sizeX, int sizeY, int sizeZ)
{
    size_t numVoxels = sizeX * sizeY * sizeZ;
    grid.resize(numVoxels);

    std::for_each(std::execution::par, grid.begin(), grid.end(),
        [&](uint8_t& voxel) {
            size_t idx = &voxel - &grid[0];

            int x = idx % sizeX;
            int y = (idx / sizeX) % sizeY;
            int z = idx / (sizeX * sizeY);

            // Floor and ceiling
            if(z == 0 || z == sizeZ - 1) { voxel = 1; return; }

            // Pillars
            if((x % 16 == 0) && (y % 16 == 0)) { voxel = 1; return; }

            // Central sphere
            float cx = sizeX / 2.0f;
            float cy = sizeY / 2.0f;
            float cz = sizeZ / 2.0f;
            float dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
            if(dist < sizeX / 8.0f) { voxel = 1; return; }

            // Noise clumps
            voxel = (noise3D(float(x), float(y), float(z)) > 0.7f) ? 1 : 0;
        });
}

const ogt_vox_scene* VoxelRenderingScene::loadVox(const char* voxFilePath) const
{
    std::ifstream file(voxFilePath, std::ios::ate | std::ios::binary);

    if(!file.is_open())
    {
        fmt::println("Could not open the file: {}", voxFilePath);
        return {};
    }

    // As the cursor is already at the end, we can directly asses the byte size of the file
    size_t bufferSize = (size_t)file.tellg();

    // Store the shader code
    std::vector<uint8_t> buffer(bufferSize);

    // Put the cursor at the beginning
    file.seekg(0);

    // Load the entire file into the buffer (read() reads the file byte by byte)
    file.read(reinterpret_cast<char*>(buffer.data()), bufferSize);

    // We are done with the file
    file.close();

    // construct the scene from the buffer
    const ogt_vox_scene* scene = ogt_vox_read_scene(buffer.data(), bufferSize);

    return scene;
}

void VoxelRenderingScene::loadData(uint32_t modelID)
{
    // Make sure that Vulkan is done working with the buffer (not the best way but in this case this scene does not do anything else other than rendering a geometry)
    vkDeviceWaitIdle(pEngine->device);
    clearBuffers();
    std::vector<uint8_t> gridData; glm::uvec3 gridSize;
    const ogt_vox_scene* pVoxScene = nullptr;

    switch(modelID)
    {
    case 0:
        pVoxScene = loadVox("../../assets/biome.vox");
        break;
    case 1:
        pVoxScene = loadVox("../../assets/monument.vox");
        break;
    case 2:
        pVoxScene = loadVox("../../assets/teapot.vox");
        break;
    default:
        // fmt::println("No existing model is selected!");
        break;
    }

    // Extract data out of vox scene
    if(pVoxScene)
    {
        // Only one model per scene, at least for now.
        const ogt_vox_model* model = pVoxScene->models[0];
        gridSize = glm::uvec3(model->size_x, model->size_y, model->size_z);
        size_t numVoxels = gridSize.x * gridSize.y * gridSize.z;
        gridData.resize(numVoxels);
        std::memcpy(gridData.data(), model->voxel_data, numVoxels);
        delete pVoxScene;
    }

    // TODO: Testing random data
    // gridSize = glm::uvec3(512);
    // generateVoxelScene(gridData, gridSize.x, gridSize.y, gridSize.z);
    // gridData.resize(gridSize.x * gridSize.y * gridSize.z);
    //fillRandomVoxelData(gridData);

    // Create the chunked version of the grid
    chunkedVolumeData = std::make_unique<ChunkedVolumeData>(pEngine, gridData, gridSize, chunkSize, gridLowerCornerPos, gridUpperCornerPos);
    gridData.clear();

    // Allocate the chunk buffer on GPU and load the whole data at the beginning only once
    size_t numChunks = chunkedVolumeData->getNumChunksFlat();
    size_t voxelChunksBufferSize = numChunks * chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(uint8_t);
    voxelChunksBuffer = pEngine->uploadStagingBuffer(chunkedVolumeData->getStagingBuffer(), voxelChunksBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    voxelChunksBufferBaseAddress = pEngine->getBufferDeviceAddress(voxelChunksBuffer.buffer);

    // Once the data is loaded staging buffer is no longer needed
    chunkedVolumeData->destroyStagingBuffer(pEngine);

    // Allocate Indirect Buffers
    const std::vector<VolumeChunk>& chunks = chunkedVolumeData->getChunks();
    std::vector<VoxelRenderingIndirectPass::ChunkMetadata> chunkMetadata(numChunks);
    {
        std::for_each(std::execution::par, chunkMetadata.begin(), chunkMetadata.end(), [&](VoxelRenderingIndirectPass::ChunkMetadata& metadata) {
            size_t i = &metadata - chunkMetadata.data();
            metadata.lowerCornerPos = chunks[i].lowerCornerPos;
            metadata.upperCornerPos = chunks[i].upperCornerPos;
            metadata.voxelBufferDeviceAddress = voxelChunksBufferBaseAddress + chunks[i].stagingBufferOffset;
        });
    }
    chunkMetadataBuffer = pEngine->createAndUploadGPUBuffer(numChunks * sizeof(VoxelRenderingIndirectPass::ChunkMetadata), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, chunkMetadata.data());

    size_t maxNumTaskInvocations = numChunks * ((chunkSize.x * chunkSize.y * chunkSize.z) / (blockSize * blockSize * blockSize));
    chunkDrawDataBuffer = pEngine->createBuffer(maxNumTaskInvocations * sizeof(VoxelRenderingIndirectPass::ChunkDrawData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // Set Active Chunk information. It is enough to set it once as all the chunks are always active in Voxel Renderer for now
    numActiveChunks = chunkedVolumeData->getNumChunksFlat();
    std::vector<uint32_t> activeChunks(numActiveChunks);
    std::iota(activeChunks.begin(), activeChunks.end(), 0);
    activeChunkIndicesBuffer = pEngine->createAndUploadGPUBuffer(numChunks * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, activeChunks.data());

    drawChunkCountBuffer = pEngine->createBuffer(3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // Set Voxel Rendering Params
    VoxelRenderingIndirectPass::SetGridShellSizes(chunkSize, chunkedVolumeData->getShellSize());
    VoxelRenderingIndirectPass::SetChunkBufferAddresses(pEngine->getBufferDeviceAddress(chunkMetadataBuffer.buffer), pEngine->getBufferDeviceAddress(chunkDrawDataBuffer.buffer), pEngine->getBufferDeviceAddress(drawChunkCountBuffer.buffer));
    VoxelRenderingIndirectPass::SetNumChunks(chunkedVolumeData->getNumChunksFlat());
    glm::vec3 voxelSize = (gridUpperCornerPos - gridLowerCornerPos) / glm::vec3(gridSize - 1u);
    VoxelRenderingIndirectPass::SetVoxelSize(voxelSize);

    // Occluder Prepass params
    OccluderPrePass::SetChunkBufferAddresses(pEngine->getBufferDeviceAddress(chunkMetadataBuffer.buffer), pEngine->getBufferDeviceAddress(activeChunkIndicesBuffer.buffer));

    // Prepare Chunk Visualization
    ChunkVisualizationPass::SetChunkBufferAddresses(pEngine->getBufferDeviceAddress(chunkMetadataBuffer.buffer), pEngine->getBufferDeviceAddress(activeChunkIndicesBuffer.buffer));
    ChunkVisualizationPass::SetNumActiveChunks(numActiveChunks);
}

void VoxelRenderingScene::clearBuffers()
{
    pEngine->destroyBuffer(voxelChunksBuffer);
    pEngine->destroyBuffer(chunkMetadataBuffer);
    pEngine->destroyBuffer(chunkDrawDataBuffer);
    pEngine->destroyBuffer(activeChunkIndicesBuffer);
    pEngine->destroyBuffer(drawChunkCountBuffer);
}
