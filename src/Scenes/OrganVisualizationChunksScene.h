#pragma once

#include "Scene.h"

#include <Pass/MarchingCubesPass.h>
#include <Pass/CircleGridPlanePass.h>
#include <Pass/ChunkVisualizationPass.h>
#include <Pass/HZBDownSamplePass.h>

#include <Data/ChunkedVolumeData.h>

class OrganVisualizationChunksScene : public Scene
{
public:
	virtual void load(VulkanEngine* engine) override;
	virtual void processSDLEvents(SDL_Event& e) override;
	virtual void handleUI() override; // Add UI field in ImGUI for scene parameters
	virtual void update() override; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) override; // called by drawing logic in engine to draw the scene
	virtual void performPreRenderPassOps(VkCommandBuffer cmd); // called before drawFrame to perform any operations that would like to be done after rendering 
	virtual void performPostRenderPassOps(VkCommandBuffer cmd); // called after drawFrame to perform any operations that would like to be done after rendering 
	virtual ~OrganVisualizationChunksScene();
private:
	void loadData(uint32_t organID);
	std::pair<std::vector<float>, glm::uvec3> loadCTheadData() const;
	std::pair<std::vector<float>, glm::uvec3> loadOrganAtlasData(const char* organPathBase) const;

	void createChunkVisualizationBuffer(const std::vector<VolumeChunk>& chunks);
	void executeMCUnsorted(VkCommandBuffer cmd) const;
	void executeMCSorted(VkCommandBuffer cmd) const;
	void executeMCLoadOnce(VkCommandBuffer cmd) const;
private:
	// Data Loading Params
	std::vector<std::string> organNames; // This is for selecting the organ data from UI. The names are hardcoded. 
	uint32_t selectedOrganID; // Keep track of the current data ID to see if the data is changed.
private:
	MarchingCubesPass::MCSettings mcSettings; // Keep track of settings to be able to modify it via GUI and update once before the render
	std::unique_ptr<ChunkedVolumeData> chunkedVolumeData;
	glm::uvec3 chunkSize = glm::uvec3(32, 32, 32);
	float minVolumeIsoValue = 0.0; float maxVolumeIsoValue = 1.0f; size_t numBins = 12;
	AllocatedBuffer voxelChunksBuffer; // a pre-determined sized buffer that holds all the chunks
	VkDeviceAddress voxelChunksBufferBaseAddress;
	AllocatedBuffer chunkVisualizationBuffer;
	VkDeviceAddress chunkVisualizationBufferAddress;
	size_t numChunksInGpu = 32;
	bool showChunks = false;
	bool executeChunksSorted = false;
	bool dataFitsInGPU = true;
};