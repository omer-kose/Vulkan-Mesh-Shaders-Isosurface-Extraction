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
	std::pair<std::vector<uint8_t>, glm::uvec3> loadCTheadData() const;
	std::pair<std::vector<uint8_t>, glm::uvec3> loadOrganAtlasData(const char* organPathBase) const;
	void clearBuffers();

	void createChunkVisualizationBuffer(const std::vector<VolumeChunk>& chunks);
private:
	// Data Loading Params
	std::vector<std::string> organNames; // This is for selecting the organ data from UI. The names are hardcoded. 
	uint32_t selectedOrganID; // Keep track of the current data ID to see if the data is changed.
private:
	glm::uvec3 gridSize;
	glm::uvec3 shellSize;
	float prevFrameIsovalue; // To keep track of change in isovalue to trigger active chunk indices update 
	float isovalue;
	std::unique_ptr<ChunkedVolumeData> chunkedVolumeData;
	glm::uvec3 chunkSize;
	AllocatedBuffer voxelChunksBuffer; // a pre-determined sized buffer that holds all the chunks
	VkDeviceAddress voxelChunksBufferBaseAddress;
	AllocatedBuffer chunkVisualizationBuffer;
	VkDeviceAddress chunkVisualizationBufferAddress;
	bool showChunks = false;
	// Indirect 
	bool indirect = true;
	AllocatedBuffer chunkMetadataBuffer;
	AllocatedBuffer chunkDrawDataBuffer;
	uint32_t numActiveChunks; // Keep track of it as it is needed for compute dispatch
	AllocatedBuffer activeChunkIndicesStagingBuffer; // when isovalue changes active indices change so for an update, I will be keeping this around
	AllocatedBuffer activeChunkIndicesBuffer;
	AllocatedBuffer drawChunkCountBuffer;
};