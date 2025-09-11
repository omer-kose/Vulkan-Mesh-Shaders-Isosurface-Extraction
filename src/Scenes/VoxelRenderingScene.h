#pragma once

#include "Scene.h"

#include <Pass/VoxelRenderingIndirectPass.h>
#include <Pass/ChunkVisualizationPass.h>
#include <Pass/HZBDownSamplePass.h>

#include <Data/ChunkedVolumeData.h>
#include <Data/ogt_vox.h>

class VoxelRenderingScene : public Scene
{
public:
	virtual void load(VulkanEngine* engine) override;
	virtual void processSDLEvents(SDL_Event& e) override;
	virtual void handleUI() override; // Add UI field in ImGUI for scene parameters
	virtual void update(float dt) override; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) override; // called by drawing logic in engine to draw the scene
	virtual void performPreRenderPassOps(VkCommandBuffer cmd); // called before drawFrame to perform any operations that would like to be done after rendering 
	virtual void performPostRenderPassOps(VkCommandBuffer cmd); // called after drawFrame to perform any operations that would like to be done after rendering 
	virtual ~VoxelRenderingScene();
private:
	void loadData(uint32_t modelID);
	void clearBuffers();

	void fillRandomVoxelData(std::vector<uint8_t>& grid, float fillProbability = 0.3f, int seed = 42);
	void generateVoxelScene(std::vector<uint8_t>& grid, int sizeX, int sizeY, int sizeZ);
	const ogt_vox_scene* loadVox(const char* voxFilePath) const;
private:
	// Data Loading Params
	std::vector<std::string> modelNames; // This is for selecting the organ data from UI. The names are hardcoded. 
	uint32_t selectedModelID; // Keep track of the current data ID to see if the data is changed.
private:
	glm::uvec3 chunkSize;
	glm::uvec3 shellSize;
	glm::vec3 gridLowerCornerPos; // in world space
	glm::vec3 gridUpperCornerPos; // in world space
	std::unique_ptr<ChunkedVolumeData> chunkedVolumeData;
	AllocatedBuffer voxelChunksBuffer; // a pre-determined sized buffer that holds all the chunks
	VkDeviceAddress voxelChunksBufferBaseAddress;
	bool showChunks = false;
	// Indirect 
	AllocatedBuffer chunkMetadataBuffer;
	AllocatedBuffer chunkDrawDataBuffer;
	uint32_t numActiveChunks; // In Voxel Renderer, all the chunks are always active, at least for now.
	AllocatedBuffer activeChunkIndicesBuffer;
	AllocatedBuffer drawChunkCountBuffer;
	// Resources
	AllocatedBuffer colorPaletteBuffer; 
private:
	// Dispatch related constants
	uint8_t blockSize;
	size_t blocksPerChunk;
};