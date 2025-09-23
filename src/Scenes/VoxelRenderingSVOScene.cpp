#include "VoxelRenderingSVOScene.h"

#include <Core/vk_engine.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_barriers.h>

#include <Util/VoxelTerrainGenerator.h>

#include <fstream>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <execution>
#include <random>

void VoxelRenderingSVOScene::load(VulkanEngine* engine)
{
    pEngine = engine;

    gridLowerCornerPos = glm::vec3(-0.5f);
    gridUpperCornerPos = glm::vec3(0.5f);

    modelNames = { "biome", "monument", "teapot", "Voxel Terrain" };

    selectedModelID = 0;
    loadData(selectedModelID);

    // Set the camera
    mainCamera = Camera(glm::vec3(0.0f, 0.0f, 2.0f), 0.0f, 0.0f);
    mainCamera.setSpeed(20.0f);

    VkExtent2D windowExtent = pEngine->getWindowExtent();
    fov = glm::radians(45.0f);
    aspectRatio = float(windowExtent.width) / float(windowExtent.height);
    LODPixelThreshold = 1.0f;

    // Set attachment clear color
    pEngine->setColorAttachmentClearColor(VkClearValue{ 0.6f, 0.9f, 1.0f, 1.0f });

    VoxelRenderingIndirectSVOPass::SetDepthPyramidBinding(pEngine, HZBDownSamplePass::GetDepthPyramidImageView(), HZBDownSamplePass::GetDepthPyramidSampler());
    VoxelRenderingIndirectSVOPass::SetDepthPyramidSizes(HZBDownSamplePass::GetDepthPyramidWidth(), HZBDownSamplePass::GetDepthPyramidHeight());
}

void VoxelRenderingSVOScene::processSDLEvents(SDL_Event& e)
{
    mainCamera.processSDLEvent(e);
}

void VoxelRenderingSVOScene::handleUI()
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
    
    ImGui::SliderFloat("LOD Pixel Threshold", &LODPixelThreshold, 1.0f, 32.0f);
    ImGui::End();
}

void VoxelRenderingSVOScene::update(float dt)
{
    mainCamera.update(dt);

    sceneData.cameraPos = mainCamera.position;
    sceneData.view = mainCamera.getViewMatrix();
    constexpr float zNear = 0.01f;
    constexpr float zFar = 10000.f;


    const float f = 1.0f / tanf(fov / 2.0f);;
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
    VoxelRenderingIndirectSVOPass::SetCameraZNear(zNear);
}

void VoxelRenderingSVOScene::drawFrame(VkCommandBuffer cmd)
{
    VoxelRenderingIndirectSVOPass::ExecuteGraphicsPass(pEngine, cmd, drawNodeCountBuffer.buffer);
}

void VoxelRenderingSVOScene::performPreRenderPassOps(VkCommandBuffer cmd)
{
    bool pixelThresholdChanged = std::fabs(prevLODPixelThreshold - LODPixelThreshold) >= std::numeric_limits<float>::epsilon();
    bool cameraDirty = mainCamera.isDirty();
    if(pixelThresholdChanged || cameraDirty)
    {
        // Fetch nodes to be processed this frame wrt camera (LOD)
        const std::vector<uint32_t> activeNodes = pSvo->selectNodesScreenSpace(mainCamera.position, fov, aspectRatio, pEngine->getWindowExtent().height, LODPixelThreshold);
        size_t numActiveNodes = activeNodes.size();
        VoxelRenderingIndirectSVOPass::SetNumActiveNodes(numActiveNodes);
        uint32_t* pStagingBuffer = (uint32_t*)pEngine->getMappedStagingBufferData(activeNodeIndicesStagingBuffer);
        std::memcpy(pStagingBuffer, activeNodes.data(), numActiveNodes * sizeof(uint32_t));

        // issue buffer copy
        VkBufferCopy copy{};
        copy.dstOffset = 0;
        copy.srcOffset = 0;
        copy.size = numActiveNodes * sizeof(uint32_t);

        vkCmdCopyBuffer(cmd, activeNodeIndicesStagingBuffer.buffer, activeNodeIndicesBuffer.buffer, 1, &copy);
        // Synchronize with a barrier
        VkBufferMemoryBarrier2 copyActiveChunksBarrier = vkutil::bufferBarrier(activeNodeIndicesBuffer.buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

        vkutil::pipelineBarrier(cmd, 0, 1, &copyActiveChunksBarrier, 0, nullptr);

        if(cameraDirty)
        {
            mainCamera.clearDirtyBit();
        }

        if(pixelThresholdChanged)
        {
            prevLODPixelThreshold = LODPixelThreshold;
        }
    }

    // Clear draw count back to 0 
    vkCmdFillBuffer(cmd, drawNodeCountBuffer.buffer, 0, 4, 0);
    vkCmdFillBuffer(cmd, drawNodeCountBuffer.buffer, 4, 8, 1); // set y and z group numbers to 1
    VkBufferMemoryBarrier2 fillBarrier = vkutil::bufferBarrier(drawNodeCountBuffer.buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    vkutil::pipelineBarrier(cmd, 0, 1, &fillBarrier, 0, nullptr);

    // Compute Culling and Preparing Indirect Commands Pass
    VoxelRenderingIndirectSVOPass::ExecuteComputePass(pEngine, cmd);
    // Synchronize and Protect drawData and drawCount buffers before indirect dispatch
    VkBufferMemoryBarrier2 cullBarriers[] = {
        vkutil::bufferBarrier(nodeDrawDataBuffer.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, VK_ACCESS_SHADER_READ_BIT),
        vkutil::bufferBarrier(drawNodeCountBuffer.buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
    };

    vkutil::pipelineBarrier(cmd, 0, 2, cullBarriers, 0, nullptr);
}

void VoxelRenderingSVOScene::performPostRenderPassOps(VkCommandBuffer cmd)
{
    HZBDownSamplePass::Execute(pEngine, cmd);
    sceneData.prevViewProj = sceneData.viewproj;
}

VoxelRenderingSVOScene::~VoxelRenderingSVOScene()
{
    clearBuffers();
}

const ogt_vox_scene* VoxelRenderingSVOScene::loadVox(const char* voxFilePath) const
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

/*
    Creates and uploads the color table. Also, sets the descriptor binding
*/
void VoxelRenderingSVOScene::createColorPaletteBuffer(const void* colorTable)
{
    // Allocate the color palette buffer. 256 maximum colors. Each color is 4 channel 1 byte
    size_t colorPaletteBufferSize = 256 * 4 * sizeof(uint8_t);
    colorPaletteBuffer = pEngine->createAndUploadGPUBuffer(colorPaletteBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, (void*)colorTable);
    // Set the descriptor set
    VoxelRenderingIndirectSVOPass::SetColorPaletteBinding(pEngine, colorPaletteBuffer.buffer, colorPaletteBufferSize);
}

void VoxelRenderingSVOScene::loadData(uint32_t modelID)
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
    case 3:
    {
        // Generate random voxel terrain
        gridSize = glm::uvec3(1024);
        gridLowerCornerPos = glm::vec3(0.0f);
        gridUpperCornerPos = glm::vec3(30.0f);
        TerrainParams params;
        params.seed = 12345;
        params.heightFrequency = 1.0f / 128.0f;
        params.heightAmplitude = 300.0f; // mountains up to ~300 voxels
        params.enableTerrace = false;
        params.enableCaves = true;
        params.enableClouds = true;
        gridData = generateVoxelTerrain(gridSize, gridLowerCornerPos, gridUpperCornerPos, params);
        // Build and assign the color palette
        const std::vector<VoxelColor> colorTable = buildTerrainColorTable(params);
        createColorPaletteBuffer((const void*)colorTable.data());
        break;
    }
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
        createColorPaletteBuffer((const void*)pVoxScene->palette.color);
        delete pVoxScene;
    }

    // Create SVO from the grid data
    pSvo = std::make_unique<SVO>(gridData, gridSize, gridLowerCornerPos, gridUpperCornerPos);
    gridData.clear();

    // Allocate the flat SVO GPU Node buffer and upload it once
    const std::vector<SVONodeGPU> gpuNodes = pSvo->getFlatGPUNodes();
    size_t svoNodeGPUBufferSize = gpuNodes.size() * sizeof(SVONodeGPU);
    svoNodeGPUBuffer = pEngine->createAndUploadGPUBuffer(svoNodeGPUBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, gpuNodes.data());

    // Allocate and upload Bricks to the gpu onces
    const std::vector<FineBrick>& fineBricks = pSvo->getFineBricks();
    size_t fineBrickBufferSize = fineBricks.size() * sizeof(FineBrick);
    fineBrickBuffer = pEngine->createAndUploadGPUBuffer(fineBrickBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, fineBricks.data());

    const std::vector<CoarseBrick>& coarseBricks = pSvo->getCoarseBricks();
    size_t coarseBrickBufferSize = coarseBricks.size() * sizeof(CoarseBrick);
    coarseBrickBuffer = pEngine->createAndUploadGPUBuffer(coarseBrickBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, coarseBricks.data());

    // TODO: Once data is upload no need to store flat gpu nodes in the memory
    // TODO UPDATE: I actually reuse flatGPUNodes to avoid recomputation of AABB's. I might store another array for lookups and delete GPU Nodes as they are larger in memory per unit
    // pSvo->clearFlatGPUNodes();
    pSvo->clearBricks();

    // Allocate Indirect Buffers
    size_t maxNumTaskInvocations = gpuNodes.size();
    nodeDrawDataBuffer = pEngine->createBuffer(maxNumTaskInvocations * sizeof(VoxelRenderingIndirectSVOPass::NodeDrawData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    drawNodeCountBuffer = pEngine->createBuffer(3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    activeNodeIndicesStagingBuffer = pEngine->createBuffer(gpuNodes.size() * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    activeNodeIndicesBuffer = pEngine->createBuffer(gpuNodes.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // Set Voxel Rendering Params
    VoxelRenderingIndirectSVOPass::SetBufferAddresses(pEngine->getBufferDeviceAddress(svoNodeGPUBuffer.buffer), pEngine->getBufferDeviceAddress(fineBrickBuffer.buffer), pEngine->getBufferDeviceAddress(coarseBrickBuffer.buffer), pEngine->getBufferDeviceAddress(nodeDrawDataBuffer.buffer),
        pEngine->getBufferDeviceAddress(drawNodeCountBuffer.buffer), pEngine->getBufferDeviceAddress(activeNodeIndicesBuffer.buffer));
    
    VoxelRenderingIndirectSVOPass::SetLeafLevel(pSvo->getLeafLevel());

    // Trigger LOD selection
    prevLODPixelThreshold = -LODPixelThreshold;
}

void VoxelRenderingSVOScene::clearBuffers()
{
    pEngine->destroyBuffer(svoNodeGPUBuffer);
    pEngine->destroyBuffer(fineBrickBuffer);
    pEngine->destroyBuffer(coarseBrickBuffer);
    pEngine->destroyBuffer(nodeDrawDataBuffer);
    pEngine->destroyBuffer(drawNodeCountBuffer);
    pEngine->destroyBuffer(activeNodeIndicesStagingBuffer);
    pEngine->destroyBuffer(activeNodeIndicesBuffer);
    pEngine->destroyBuffer(colorPaletteBuffer);
}
