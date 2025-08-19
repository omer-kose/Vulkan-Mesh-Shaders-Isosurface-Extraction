#include "OrganVisualizationScene.h"

#include <Core/vk_engine.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>
#include <Core/vk_barriers.h>

#include <fstream>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

void OrganVisualizationChunksScene::load(VulkanEngine* engine)
{
    pEngine = engine;

    voxelDataImageSampler = engine->createImageSampler(VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_REDUCTION_MODE_MIN);

    // Set Grid Plane Pass Settings
    CircleGridPlanePass::SetPlaneHeight(-0.1f);

    // Set the camera
    mainCamera = Camera(glm::vec3(-2.0f, 0.0f, 2.0f), 0.0f, -45.0f);
    mainCamera.setSpeed(0.02f);

    // Set attachment clear color
    pEngine->setColorAttachmentClearColor(VkClearValue{ 0.6f, 0.9f, 1.0f, 1.0f });

    MarchingCubesPass::SetDepthPyramidBinding(pEngine, HZBDownSamplePass::GetDepthPyramidImageView(), HZBDownSamplePass::GetDepthPyramidSampler());
    MarchingCubesPass::SetDepthPyramidSizes(HZBDownSamplePass::GetDepthPyramidWidth(), HZBDownSamplePass::GetDepthPyramidHeight());

    MarchingCubesIndirectPass::SetDepthPyramidBinding(pEngine, HZBDownSamplePass::GetDepthPyramidImageView(), HZBDownSamplePass::GetDepthPyramidSampler());
    MarchingCubesIndirectPass::SetDepthPyramidSizes(HZBDownSamplePass::GetDepthPyramidWidth(), HZBDownSamplePass::GetDepthPyramidHeight());

    organNames = { "CThead", "Kidney" };

    selectedOrganID = 0;
    loadData(selectedOrganID);
}

void OrganVisualizationChunksScene::processSDLEvents(SDL_Event& e)
{
    mainCamera.processSDLEvent(e);
}

void OrganVisualizationChunksScene::handleUI()
{
    ImGui::Begin("Marching Cubes Parameters");

    // Organ Selection
    if(ImGui::BeginCombo("Scene Selection", organNames[selectedOrganID].c_str()))
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

    ImGui::SliderFloat("Iso Value", &isovalue, 0.0f, 1.0f);
    if(ImGui::Checkbox("Indirect Dispatch", &indirect))
    {
        if(indirect)
        {
            prevFrameIsovalue = -isovalue; // invalidate prev isovalue to trigger the first active chunk indices load 
        }
    }
    ImGui::Checkbox("Show Chunks", &showChunks);
    if(ImGui::Checkbox("Use Data Image", &useImageData))
    {
        MarchingCubesPass::SetDataSourceImageFlag(useImageData);
    }
    ImGui::End();
}

void OrganVisualizationChunksScene::update()
{
    mainCamera.update();

    sceneData.view = mainCamera.getViewMatrix();
    constexpr float fov = glm::radians(45.0f);
    constexpr float zNear = 0.01f;
    constexpr float zFar = 10000.f;

    VkExtent2D windowExtent = pEngine->getWindowExtent();

    // sceneData.proj = glm::perspectiveRH_ZO(fov, (float)windowExtent.width / (float)windowExtent.height, zFar, zNear); // reverting zFar and zNear as I use a inverted depth buffer

    const float f = 1.0f / tanf(fov / 2.0f);;
    const float aspectRatio = float(windowExtent.width) / float(windowExtent.height);
    sceneData.proj = glm::mat4(
        f / aspectRatio, 0.0f, 0.0f, 0.0f,
        0.0f, -f, 0.0f, 0.0f, // inverting y inplace for Vulkan's convention
        0.0f, 0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, zNear, 0.0f
    );

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis
    //sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(0.1f);
    sceneData.sunlightColor = glm::vec4(1.0f);
    glm::vec3 directionalLightDir = glm::normalize(glm::vec3(0.0f, -1.0f, -0.5f));
    sceneData.sunlightDirection = glm::vec4(directionalLightDir, 1.0f);

    // Update the MC params (cheap operation but could be checked if there is any change)
    MarchingCubesPass::SetInputIsovalue(isovalue);
    MarchingCubesPass::SetCameraZNear(zNear);

    MarchingCubesIndirectPass::SetInputIsovalue(isovalue);
    MarchingCubesIndirectPass::SetCameraZNear(zNear);

    ChunkVisualizationPass::SetInputIsovalue(isovalue);
}

void OrganVisualizationChunksScene::drawFrame(VkCommandBuffer cmd)
{
    CircleGridPlanePass::Execute(pEngine, cmd);
    if(showChunks)
    {
        ChunkVisualizationPass::Execute(pEngine, cmd, chunkedVolumeData->getNumChunksFlat(), 3.0f);
    }
    
    if(indirect)
    {
        MarchingCubesIndirectPass::ExecuteGraphicsPass(pEngine, cmd, drawChunkCountBuffer.buffer);
    }
    else
    {
        // Fetch the chunks that contain the input iso-value in range
        std::vector<VolumeChunk*> renderChunks = chunkedVolumeData->query(isovalue);
        int numRenderChunks = renderChunks.size();

        for(int i = 0; i < numRenderChunks; ++i)
        {
            // Dispatch the mesh shaders for each chunk
            MarchingCubesPass::SetVoxelBufferDeviceAddress(voxelChunksBufferBaseAddress + renderChunks[i]->stagingBufferOffset);
            MarchingCubesPass::SetChunkStartIndex(renderChunks[i]->startIndex);
            MarchingCubesPass::SetGridCornerPositions(renderChunks[i]->lowerCornerPos, renderChunks[i]->upperCornerPos);
            MarchingCubesPass::Execute(pEngine, cmd);

        }
    }
}

void OrganVisualizationChunksScene::performPreRenderPassOps(VkCommandBuffer cmd)
{
    if(indirect)
    {
        if(std::fabs(prevFrameIsovalue - isovalue) >= std::numeric_limits<float>::epsilon())
        {
            // Update the staging buffer content 
            std::vector<VolumeChunk*> renderChunks = chunkedVolumeData->query(isovalue);
            numActiveChunks = renderChunks.size();
            MarchingCubesIndirectPass::SetNumActiveChunks(numActiveChunks);
            uint32_t* pStagingBuffer = (uint32_t*)pEngine->getMappedStagingBufferData(activeChunkIndicesStagingBuffer);
            for(int i = 0; i < numActiveChunks; ++i)
            {
                pStagingBuffer[i] = renderChunks[i]->chunkFlatIndex;
            }

            // issue buffer copy
            VkBufferCopy copy{};
            copy.dstOffset = 0;
            copy.srcOffset = 0;
            copy.size = numActiveChunks * sizeof(uint32_t);

            vkCmdCopyBuffer(cmd, activeChunkIndicesStagingBuffer.buffer, activeChunkIndicesBuffer.buffer, 1, &copy);
            // Synchronize with a barrier
            VkBufferMemoryBarrier2 copyActiveChunksBarrier = vkutil::bufferBarrier(activeChunkIndicesBuffer.buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

            vkutil::pipelineBarrier(cmd, 0, 1, &copyActiveChunksBarrier, 0, nullptr);

            prevFrameIsovalue = isovalue;
        }

        // Clear draw count back to 0 
        vkCmdFillBuffer(cmd, drawChunkCountBuffer.buffer, 0, 4, 0);
        vkCmdFillBuffer(cmd, drawChunkCountBuffer.buffer, 4, 8, 1); // set y and z group numbers to 1
        VkBufferMemoryBarrier2 fillBarrier = vkutil::bufferBarrier(drawChunkCountBuffer.buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        
        vkutil::pipelineBarrier(cmd, 0, 1, &fillBarrier, 0, nullptr);

        // Compute Culling and Preparing Indirect Commands Pass
        MarchingCubesIndirectPass::ExecuteComputePass(pEngine, cmd, numActiveChunks);
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
}

void OrganVisualizationChunksScene::performPostRenderPassOps(VkCommandBuffer cmd)
{
    HZBDownSamplePass::Execute(pEngine, cmd);
}

OrganVisualizationChunksScene::~OrganVisualizationChunksScene()
{
    clearBuffers();
    pEngine->destroyImage(voxelDataImage);
    vkDestroySampler(pEngine->device, voxelDataImageSampler, nullptr);
}

void OrganVisualizationChunksScene::createChunkVisualizationBuffer(const std::vector<VolumeChunk>& chunks)
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

void OrganVisualizationChunksScene::loadData(uint32_t organID)
{
    // Make sure that Vulkan is done working with the buffer (not the best way but in this case this scene does not do anything else other than rendering a geometry)
    vkDeviceWaitIdle(pEngine->device);
    clearBuffers();
    std::vector<float> gridData; glm::uvec3 gridSize;

    switch(organID)
    {
        case 0:
            std::tie(gridData, gridSize) = loadCTheadData();
            break;
        case 1:
            std::tie(gridData, gridSize) = loadOrganAtlasData("../../assets/organ_atlas/kidney");
            break;
        default:
            fmt::println("No existing organ id is selected!");
            break;
    }

    // Create the chunked version of the grid
    chunkedVolumeData = std::make_unique<ChunkedVolumeData>(pEngine, gridData, gridSize, chunkSize, glm::vec3(-0.5f), glm::vec3(0.5f));
    // Create Image version of the data. Image does not need chunked data as it will hold the whole data at once. Fetching will be done via an offset in the chunk
    voxelDataImage = pEngine->createImage(gridData.data(), VkExtent3D{ gridSize.x, gridSize.y, gridSize.z }, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false);

    gridData.clear();

    // Allocate the chunk buffer on GPU and load the whole data at the beginning only once
    size_t numChunks = chunkedVolumeData->getNumChunksFlat();
    size_t voxelChunksBufferSize = numChunks * chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(float);
    voxelChunksBuffer = pEngine->uploadStagingBuffer(chunkedVolumeData->getStagingBuffer(), voxelChunksBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    voxelChunksBufferBaseAddress = pEngine->getBufferDeviceAddress(voxelChunksBuffer.buffer);

   
    // Once the data is loaded staging buffer is no longer needed
    chunkedVolumeData->destroyStagingBuffer(pEngine);

    gridSize = chunkSize;
    shellSize = chunkedVolumeData->getShellSize();
    isovalue = 0.5f;
    prevFrameIsovalue = -isovalue; // something not equal to isovalue to trigger the first active chunk indices update
    MarchingCubesPass::SetGridShellSizes(gridSize, shellSize);
    MarchingCubesPass::SetInputIsovalue(isovalue);
    MarchingCubesPass::SetVoxelDataImageBinding(pEngine, voxelDataImage.imageView, voxelDataImageSampler);

    // Allocate Indirect Buffers
    const std::vector<VolumeChunk>& chunks = chunkedVolumeData->getChunks();
    std::vector<MarchingCubesIndirectPass::ChunkMetadata> chunkMetadata(numChunks);
    for(int i = 0; i < numChunks; ++i)
    {
        chunkMetadata[i].lowerCornerPos = chunks[i].lowerCornerPos;
        chunkMetadata[i].upperCornerPos = chunks[i].upperCornerPos;
        chunkMetadata[i].voxelBufferDeviceAddress = voxelChunksBufferBaseAddress + chunks[i].stagingBufferOffset;
    }
    chunkMetadataBuffer = pEngine->createAndUploadGPUBuffer(numChunks * sizeof(MarchingCubesIndirectPass::ChunkMetadata), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, chunkMetadata.data());
    
    glm::uvec3 chunkSize = chunkedVolumeData->getChunkSize();
    uint32_t blockSize = 4; // The size of the blocks that task shader processes a chunk. Hardcoding it for now
    size_t maxNumTaskInvocations = numChunks * ((chunkSize.x * chunkSize.y * chunkSize.z) / (blockSize * blockSize * blockSize));
    chunkDrawDataBuffer = pEngine->createBuffer(maxNumTaskInvocations * sizeof(MarchingCubesIndirectPass::ChunkDrawData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    activeChunkIndicesStagingBuffer = pEngine->createBuffer(numChunks * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    activeChunkIndicesBuffer = pEngine->createBuffer(numChunks * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    
    drawChunkCountBuffer = pEngine->createBuffer(3 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    // Set Marching Cubes Indirect Pass parameters
    MarchingCubesIndirectPass::SetGridShellSizes(gridSize, shellSize);
    MarchingCubesIndirectPass::SetInputIsovalue(isovalue);
    MarchingCubesIndirectPass::SetChunkBufferAddresses(pEngine->getBufferDeviceAddress(chunkMetadataBuffer.buffer), pEngine->getBufferDeviceAddress(chunkDrawDataBuffer.buffer), pEngine->getBufferDeviceAddress(activeChunkIndicesBuffer.buffer), pEngine->getBufferDeviceAddress(drawChunkCountBuffer.buffer));
    
    // Prepare Chunk Visualization
    createChunkVisualizationBuffer(chunkedVolumeData->getChunks());
    ChunkVisualizationPass::SetChunkBufferDeviceAddress(chunkVisualizationBufferAddress);
    ChunkVisualizationPass::SetInputIsovalue(isovalue);
}

std::pair<std::vector<float>, glm::uvec3> OrganVisualizationChunksScene::loadCTheadData() const
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
    VkPipelineLayout converterPipelineLayout; VkPipeline converterPipeline; ComputePipelineBuilder pipelineBuilder;
    VkPushConstantRange converterPCRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(VolumeDataConverterPushConstants) };
    VkShaderModule converterComputeShader;
    if(!vkutil::loadShaderModule(pEngine->device, "../../shaders/glsl/volume_data_convert/volume_data_convert_comp.spv", &converterComputeShader))
    {
        fmt::println("Volume Data Converter Compute Shader could not be loaded!");
    }
    std::tie(converterPipelineLayout, converterPipeline) = pipelineBuilder.buildPipeline(pEngine->device, converterComputeShader, {converterPCRange});

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

    // Download the loaded and converted grid back to extract chunks
    std::vector<float> gridData(gridSize.x * gridSize.y * gridSize.z);
    AllocatedBuffer voxelBufferCPU = pEngine->downloadGPUBuffer(voxelBuffer.buffer, voxelBufferSize);
    float* pGridData = (float*)pEngine->getMappedStagingBufferData(voxelBufferCPU);
    memcpy(gridData.data(), pGridData, voxelBufferSize);
    pEngine->destroyBuffer(voxelBuffer);
    pEngine->destroyBuffer(voxelBufferCPU);

    // During load I align CThead with right handed standard coordinate system. For that y and z axes change.
    return { gridData, glm::uvec3(gridSize.x, gridSize.z, gridSize.y) };
}

std::pair<std::vector<float>, glm::uvec3> OrganVisualizationChunksScene::loadOrganAtlasData(const char* organPathBase) const
{
    // All the data in organ atlas has the same signature
    std::string binPath = std::string(organPathBase) + ".bin";
    std::string gridSizePath = std::string(organPathBase) + "_shape.txt";

    // Load the grid size
    glm::uvec3 gridSize;
    std::ifstream file(gridSizePath);
    if(!file)
    {
        fmt::println("Could not open the file: {}", gridSizePath);
    }
    uint32_t size; int i = 0;
    while(file >> size)
    {
        gridSize[i++] = size;
    }
    file.close();

    // Read the binary grid data
    file.open(binPath, std::ios::binary);
    if(!file)
    {
        fmt::println("Could not open the file: {}", binPath);
    }
    size_t numElements = gridSize.x * gridSize.y * gridSize.z;
    std::vector<float> buffer(numElements);
    file.read(reinterpret_cast<char*>(buffer.data()), numElements * sizeof(float));

    file.close();

    return { buffer, gridSize };
}

void OrganVisualizationChunksScene::clearBuffers()
{
    pEngine->destroyBuffer(voxelChunksBuffer);
    pEngine->destroyBuffer(chunkMetadataBuffer);
    pEngine->destroyBuffer(chunkDrawDataBuffer);
    pEngine->destroyBuffer(activeChunkIndicesStagingBuffer);
    pEngine->destroyBuffer(activeChunkIndicesBuffer);
    pEngine->destroyBuffer(drawChunkCountBuffer);
    pEngine->destroyBuffer(chunkVisualizationBuffer);
}
