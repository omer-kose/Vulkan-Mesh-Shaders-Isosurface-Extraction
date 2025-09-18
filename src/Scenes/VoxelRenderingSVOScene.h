#pragma once

#include "Scene.h"

#include <Pass/VoxelRenderingIndirectSVOPass.h>
#include <Pass/HZBDownSamplePass.h>

#include <Data/SVO.h>
#include <Data/ogt_vox.h>

class VoxelRenderingSVOScene : public Scene
{
public:
	virtual void load(VulkanEngine* engine) override;
	virtual void processSDLEvents(SDL_Event& e) override;
	virtual void handleUI() override; // Add UI field in ImGUI for scene parameters
	virtual void update(float dt) override; // called in engine update
	virtual void drawFrame(VkCommandBuffer cmd) override; // called by drawing logic in engine to draw the scene
	virtual void performPreRenderPassOps(VkCommandBuffer cmd); // called before drawFrame to perform any operations that would like to be done after rendering 
	virtual void performPostRenderPassOps(VkCommandBuffer cmd); // called after drawFrame to perform any operations that would like to be done after rendering 
	virtual ~VoxelRenderingSVOScene();
private:
	void loadData(uint32_t modelID);
	void clearBuffers();
	
	const ogt_vox_scene* loadVox(const char* voxFilePath) const;
	void createColorPaletteBuffer(const void* colorTable);
private:
	// Data Loading Params
	std::vector<std::string> modelNames; // This is for selecting the organ data from UI. The names are hardcoded. 
	uint32_t selectedModelID; // Keep track of the current data ID to see if the data is changed.
private:
	glm::vec3 gridLowerCornerPos; // in world space
	glm::vec3 gridUpperCornerPos; // in world space
	std::unique_ptr<SVO> pSvo;
	AllocatedBuffer svoNodeGPUBuffer; // flattened SVO GPU Nodes
	AllocatedBuffer brickBuffer; 
	// Indirect 
	AllocatedBuffer nodeDrawDataBuffer;
	AllocatedBuffer drawNodeCountBuffer;
	AllocatedBuffer activeNodeIndicesStagingBuffer; // when camera moves active nodes change due to LOD
	AllocatedBuffer activeNodeIndicesBuffer;
	// Resources
	AllocatedBuffer colorPaletteBuffer; 
	// Properties
	float fov;
	float aspectRatio;
	float LODPixelThreshold;
	float prevLODPixelThreshold; // to trigger LOD selection
};