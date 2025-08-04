#include "OrganVisualizationChunksScene.h"

#include <Core/vk_engine.h>
#include <Core/vk_initializers.h>
#include <Core/vk_pipelines.h>

#include <fstream>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

void insertTransferToMeshShaderBarrier(
    VkCommandBuffer commandBuffer,
    VkBuffer buffer,
    VkDeviceSize offset = 0,
    VkDeviceSize size = VK_WHOLE_SIZE
)
{
    VkBufferMemoryBarrier bufferBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = offset,
        .size = size
    };

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,          // Source: Transfer
        VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,    // Dest: Mesh shader
        0,
        0, nullptr,       // No memory barriers
        1, &bufferBarrier,
        0, nullptr
    );
}

void insertMeshShaderToTransferBarrier(
    VkCommandBuffer commandBuffer,
    VkBuffer buffer,
    VkDeviceSize offset = 0,
    VkDeviceSize size = VK_WHOLE_SIZE
)
{
    VkBufferMemoryBarrier bufferBarrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = offset,
        .size = size
    };
    
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,    // Source: Mesh shader
        VK_PIPELINE_STAGE_TRANSFER_BIT,           // Dest: Transfer
        0,
        0, nullptr,       // No memory barriers
        1, &bufferBarrier,
        0, nullptr
    );
}

void OrganVisualizationChunksScene::load(VulkanEngine* engine)
{
    pEngine = engine;

    organNames = { "CThead", "Kidney" };

    loadData(0);

    // Set Grid Plane Pass Settings
    CircleGridPlanePass::SetPlaneHeight(-0.1f);

    // Prepare Chunk Visualization
    createChunkVisualizationBuffer(chunkedVolumeData->getChunks());
    ChunkVisualizationPass::SetChunkBufferDeviceAddress(chunkVisualizationBufferAddress);
    ChunkVisualizationPass::SetInputIsoValue(mcSettings.isoValue);

    // Set the camera
    mainCamera = Camera(glm::vec3(-2.0f, 0.0f, 2.0f), 0.0f, -45.0f);
    mainCamera.setSpeed(0.02f);

    // Set attachment clear color
    pEngine->setColorAttachmentClearColor(VkClearValue{ 0.6f, 0.9f, 1.0f, 1.0f });

    MarchingCubesPass::SetDepthPyramidBinding(pEngine, HZBDownSamplePass::GetDepthPyramidImageView(), HZBDownSamplePass::GetDepthPyramidSampler());
    MarchingCubesPass::SetDepthPyramidSizes(HZBDownSamplePass::GetDepthPyramidWidth(), HZBDownSamplePass::GetDepthPyramidHeight());
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

    ImGui::SliderFloat("Iso Value", &mcSettings.isoValue, 0.0f, 1.0f);
    ImGui::Checkbox("Show Chunks", &showChunks);
    ImGui::Checkbox("Execute Chunks Sorted", &executeChunksSorted);
    ImGui::End();
}

void OrganVisualizationChunksScene::update()
{
    mainCamera.update();

    sceneData.view = mainCamera.getViewMatrix();
    constexpr float fov = glm::radians(45.0f);
    float zNear = 0.1f;
    float zFar = 10000.f;

    VkExtent2D windowExtent = pEngine->getWindowExtent();

    sceneData.proj = glm::perspectiveRH_ZO(fov, (float)windowExtent.width / (float)windowExtent.height, zFar, zNear); // reverting zFar and zNear as I use a inverted depth buffer

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
    MarchingCubesPass::SetCameraZNear(zNear);

    ChunkVisualizationPass::SetInputIsoValue(mcSettings.isoValue);
}

void OrganVisualizationChunksScene::drawFrame(VkCommandBuffer cmd)
{
    CircleGridPlanePass::Execute(pEngine, cmd);
    if(showChunks)
    {
        ChunkVisualizationPass::Execute(pEngine, cmd, chunkedVolumeData->getNumChunksFlat(), 3.0f);
    }
    

    if(dataFitsInGPU)
    {
        executeMCLoadOnce(cmd);
    }
    else
    {
        executeChunksSorted ? executeMCSorted(cmd) : executeMCUnsorted(cmd);
    }
}

void OrganVisualizationChunksScene::performPreRenderPassOps(VkCommandBuffer cmd)
{
}

void OrganVisualizationChunksScene::performPostRenderPassOps(VkCommandBuffer cmd)
{
    HZBDownSamplePass::Execute(pEngine, cmd);
}

OrganVisualizationChunksScene::~OrganVisualizationChunksScene()
{
    pEngine->destroyBuffer(voxelChunksBuffer);
    pEngine->destroyBuffer(chunkVisualizationBuffer);
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
    pEngine->destroyBuffer(voxelChunksBuffer);
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
    gridData.clear();
    //chunkedVolumeData->computeChunkIsoValueHistograms(minVolumeIsoValue, maxVolumeIsoValue, numBins);

    if(dataFitsInGPU)
    {
        // Allocate the chunk buffer on GPU and load the whole data at the beginning only once
        size_t voxelChunksBufferSize = chunkedVolumeData->getNumChunksFlat() * chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(float);
        voxelChunksBuffer = pEngine->uploadStagingBuffer(chunkedVolumeData->getStagingBuffer(), voxelChunksBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        voxelChunksBufferBaseAddress = pEngine->getBufferDeviceAddress(voxelChunksBuffer.buffer);
    }
    else
    {
        // Allocate the chunk buffer on GPU
        size_t voxelChunksBufferSize = numChunksInGpu * chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(float);
        voxelChunksBuffer = pEngine->createBuffer(voxelChunksBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        voxelChunksBufferBaseAddress = pEngine->getBufferDeviceAddress(voxelChunksBuffer.buffer);
    }

    mcSettings.gridSize = chunkSize;
    mcSettings.shellSize = chunkedVolumeData->getShellSize();
    mcSettings.isoValue = 0.5f;
    MarchingCubesPass::UpdateMCSettings(mcSettings);

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

void OrganVisualizationChunksScene::executeMCUnsorted(VkCommandBuffer cmd) const
{
    // Fetch the chunks that contain the input iso-value in range
    std::vector<VolumeChunk*> renderChunks = chunkedVolumeData->query(mcSettings.isoValue);
    int numRenderChunks = renderChunks.size();
    int numBatches = (numRenderChunks + numChunksInGpu - 1) / numChunksInGpu;
    // Precompute the values that will be needed for buffer upload
    VkBuffer chunksStagingBuffer = chunkedVolumeData->getStagingBuffer();
    glm::uvec3 chunkSize = chunkedVolumeData->getChunkSize();
    size_t chunkSizeInBytes = chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(float);
    std::vector<VkBufferCopy> copyRegions(numChunksInGpu); // allocating the maximum size will be reused by all the batches

    // Vulkan strictly forbids transfer operations in a render-pass so, I will end the render-pass before each transfer and begin after the operation. The contents of the drawImage should not be cleared.
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(pEngine->drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info_preserve(pEngine->depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(pEngine->drawExtent, &colorAttachment, &depthAttachment);

    for(int i = 0; i < numBatches; ++i)
    {
        int firstChunkIdx = i * numChunksInGpu;
        int numChunksToBeProcessed = std::min(numRenderChunks, (int)numChunksInGpu);

        vkCmdEndRendering(cmd);

        // Upload the chunks in the batch
        for(int c = 0; c < numChunksToBeProcessed; ++c)
        {
            VkBufferCopy copy{};
            copy.dstOffset = c * chunkSizeInBytes;
            copy.srcOffset = renderChunks[firstChunkIdx + c]->stagingBufferOffset;
            copy.size = chunkSizeInBytes;
            copyRegions[c] = copy;
        }
        vkCmdCopyBuffer(cmd, chunksStagingBuffer, voxelChunksBuffer.buffer, numChunksToBeProcessed, copyRegions.data());

        // Pipeline Barrier between buffer transfer and mesh shader dispatch
        insertTransferToMeshShaderBarrier(cmd, voxelChunksBuffer.buffer, 0, numChunksInGpu * chunkSizeInBytes);

        vkCmdBeginRendering(cmd, &renderInfo);

        // Dispatch the mesh shaders for each chunk
        for(int c = 0; c < numChunksToBeProcessed; ++c)
        {
            MarchingCubesPass::SetVoxelBufferDeviceAddress(voxelChunksBufferBaseAddress + c * chunkSizeInBytes);
            MarchingCubesPass::SetGridCornerPositions(renderChunks[firstChunkIdx + c]->lowerCornerPos, renderChunks[firstChunkIdx + c]->upperCornerPos);
            MarchingCubesPass::Execute(pEngine, cmd);
        }

        // Pipeline Barrier between mesh shader dispatch and the next buffer transfer
        if(i != numBatches - 1)
        {
            vkCmdEndRendering(cmd);
            insertMeshShaderToTransferBarrier(cmd, voxelChunksBuffer.buffer, 0, numChunksInGpu * chunkSizeInBytes);
            vkCmdBeginRendering(cmd, &renderInfo);
        }

        numRenderChunks -= numChunksInGpu;
    }
}

void OrganVisualizationChunksScene::executeMCSorted(VkCommandBuffer cmd) const
{
    // Fetch the chunks that contain the input iso-value in range
    std::vector<VolumeChunk*> renderChunks = chunkedVolumeData->query(mcSettings.isoValue);
    int numRenderChunks = renderChunks.size();
    int numBatches = (numRenderChunks + numChunksInGpu - 1) / numChunksInGpu;
    // Precompute the values that will be needed for buffer upload
    VkBuffer chunksStagingBuffer = chunkedVolumeData->getStagingBuffer();
    glm::uvec3 chunkSize = chunkedVolumeData->getChunkSize();
    size_t chunkSizeInBytes = chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(float);
    std::vector<VkBufferCopy> copyRegions(numChunksInGpu); // allocating the maximum size will be reused by all the batches

    // Sort the chunks (actually indices of them) wrt number of estimate triangles they will produce. So that, in each batch dispatch we have a more uniform execution time until the barrier
    std::vector<uint32_t> sortedChunkIndices(renderChunks.size());
    for(uint32_t i = 0; i < renderChunks.size(); ++i)
    {
        sortedChunkIndices[i] = i;
    }

    std::sort(sortedChunkIndices.begin(), sortedChunkIndices.end(), [&](uint32_t iA, uint32_t iB){
        size_t numTrianglesA = chunkedVolumeData->estimateNumTriangles(*renderChunks[iA], mcSettings.isoValue);
        size_t numTrianglesB = chunkedVolumeData->estimateNumTriangles(*renderChunks[iB], mcSettings.isoValue);

        return numTrianglesA > numTrianglesB;
    });

    // Vulkan strictly forbids transfer operations in a render-pass so, I will end the render-pass before each transfer and begin after the operation. The contents of the drawImage should not be cleared.
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(pEngine->drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info_preserve(pEngine->depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(pEngine->drawExtent, &colorAttachment, &depthAttachment);

    for(int i = 0; i < numBatches; ++i)
    {
        int firstChunkIdx = i * numChunksInGpu;
        int numChunksToBeProcessed = std::min(numRenderChunks, (int)numChunksInGpu);

        vkCmdEndRendering(cmd);

        // Upload the chunks in the batch
        for(int c = 0; c < numChunksToBeProcessed; ++c)
        {
            VkBufferCopy copy{};
            copy.dstOffset = c * chunkSizeInBytes;
            copy.srcOffset = renderChunks[sortedChunkIndices[firstChunkIdx + c]]->stagingBufferOffset;
            copy.size = chunkSizeInBytes;
            copyRegions[c] = copy;
        }
        vkCmdCopyBuffer(cmd, chunksStagingBuffer, voxelChunksBuffer.buffer, numChunksToBeProcessed, copyRegions.data());

        // Pipeline Barrier between buffer transfer and mesh shader dispatch
        insertTransferToMeshShaderBarrier(cmd, voxelChunksBuffer.buffer);

        vkCmdBeginRendering(cmd, &renderInfo);

        // Dispatch the mesh shaders for each chunk
        for(int c = 0; c < numChunksToBeProcessed; ++c)
        {
            MarchingCubesPass::SetVoxelBufferDeviceAddress(voxelChunksBufferBaseAddress + c * chunkSizeInBytes);
            MarchingCubesPass::SetGridCornerPositions(renderChunks[sortedChunkIndices[firstChunkIdx + c]]->lowerCornerPos, renderChunks[sortedChunkIndices[firstChunkIdx + c]]->upperCornerPos);
            MarchingCubesPass::Execute(pEngine, cmd);
        }

        // Pipeline Barrier between mesh shader dispatch and the next buffer transfer
        if(i != numBatches - 1)
        {
            vkCmdEndRendering(cmd);
            insertMeshShaderToTransferBarrier(cmd, voxelChunksBuffer.buffer);
            vkCmdBeginRendering(cmd, &renderInfo);
        }

        numRenderChunks -= numChunksInGpu;
    }
}

void OrganVisualizationChunksScene::executeMCLoadOnce(VkCommandBuffer cmd) const
{
    // Fetch the chunks that contain the input iso-value in range
    std::vector<VolumeChunk*> renderChunks = chunkedVolumeData->query(mcSettings.isoValue);
    int numRenderChunks = renderChunks.size();
    size_t chunkSizeInBytes = chunkedVolumeData->getTotalNumPointsPerChunk() * sizeof(float);

    for(int i = 0; i < numRenderChunks; ++i)
    {
        // Dispatch the mesh shaders for each chunk
        MarchingCubesPass::SetVoxelBufferDeviceAddress(voxelChunksBufferBaseAddress + renderChunks[i]->stagingBufferOffset);
        MarchingCubesPass::SetGridCornerPositions(renderChunks[i]->lowerCornerPos, renderChunks[i]->upperCornerPos);
        MarchingCubesPass::Execute(pEngine, cmd);

    }
}
